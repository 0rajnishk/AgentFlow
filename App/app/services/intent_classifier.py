# app/services/intent_classifier.py

import google.generativeai as genai
from app.core.config import get_settings
from app.core.logging import logger
from enum import Enum
from typing import Dict, Any, Optional, List
import json
import time
import hashlib
from pydantic import BaseModel, Field, validator
import re

class QueryType(Enum):
    DOCUMENT_ONLY = "document_only"
    DATABASE_ONLY = "database_only"
    HYBRID = "hybrid"
    UNCLEAR = "unclear"

class PermissionType(Enum):
    FINANCE_ACCESS = "finance_access"
    PLANNING_ACCESS = "planning_access"
    ADMIN_ACCESS = "admin_access"
    REGION_ACCESS = "region_access"
    BASIC_DATABASE_ACCESS = "basic_database_access"
    DOCUMENT_ACCESS = "document_access"

class ClassificationResult(BaseModel):
    """Pydantic model for type-safe classification results"""
    query_type: str = Field(..., pattern="^(document_only|database_only|hybrid|unclear)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=10, max_length=500)
    keywords: List[str] = Field(default_factory=list)
    suggested_sources: List[str] = Field(default_factory=list)
    required_permissions: List[str] = Field(default_factory=list)
    
    @validator('suggested_sources')
    def validate_sources(cls, v):
        valid_sources = {'documents', 'database'}  # Removed 'both' to prevent validation errors
        for source in v:
            if source not in valid_sources:
                raise ValueError(f"Invalid source: {source}")
        return v
    
    @validator('required_permissions')
    def validate_permissions(cls, v):
        valid_permissions = {
            'finance_access', 'planning_access', 'admin_access', 
            'basic_database_access', 'document_access'
        }
        for perm in v:
            if not perm.startswith('region_access:') and perm not in valid_permissions:
                raise ValueError(f"Invalid permission: {perm}")
        return v

class PermissionAnalyzer:
    """Separate service for analyzing permission requirements"""
    
    def __init__(self):
        self.financial_keywords = {
            'profit', 'margin', 'cost', 'revenue', 'sales', 'financial', 
            'p&l', 'profit and loss', 'budget', 'expense', 'pricing'
        }
        self.planning_keywords = {
            'forecast', 'planning', 'inventory', 'capacity', 'demand', 
            'supply', 'logistics', 'warehouse', 'stock'
        }
        self.regional_patterns = [
            r'\bin\s+([A-Za-z\s]+?)(?:\s+region|\s+area|\s+zone|$)',
            r'([A-Za-z\s]+?)\s+region',
            r'from\s+([A-Za-z\s]+?)(?:\s+area|\s+zone|$)'
        ]
    
    def analyze_permissions(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Analyze query to determine required permissions"""
        permissions = []
        query_lower = query.lower()
        
        # Check for financial data access
        if any(keyword in query_lower for keyword in self.financial_keywords):
            permissions.append('finance_access')
        
        # Check for planning data access
        if any(keyword in query_lower for keyword in self.planning_keywords):
            permissions.append('planning_access')
        
        # Check for regional restrictions
        regions = self._extract_regions(query)
        for region in regions:
            permissions.append(f'region_access:{region}')
        
        # Default permissions for data access
        if not permissions:
            if any(word in query_lower for word in ['data', 'show', 'list', 'count', 'total', 'how many']):
                permissions.append('basic_database_access')
            if any(word in query_lower for word in ['policy', 'procedure', 'guideline', 'rule']):
                permissions.append('document_access')
        
        return permissions
    
    def _extract_regions(self, query: str) -> List[str]:
        """Extract region names from query using regex patterns"""
        regions = []
        for pattern in self.regional_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                region = match.strip().title()
                if len(region) > 2 and region not in regions:
                    regions.append(region)
        return regions

class QueryClassificationService:
    """Main classification service with caching and error handling"""
    
    def __init__(self):
        self.model = self._initialize_model()
        self.permission_analyzer = PermissionAnalyzer()
        self._classification_cache = {}
        self._cache_ttl = 3600  # 1 hour cache TTL
        
    def _initialize_model(self) -> genai.GenerativeModel:
        """Initialize Gemini model with error handling"""
        try:
            genai.configure(api_key=get_settings().GENAI_API_KEY)
            return genai.GenerativeModel(get_settings().GENAI_MODEL)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def _generate_cache_key(self, query: str, user_context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for query and context"""
        context_str = json.dumps(user_context or {}, sort_keys=True)
        combined = f"{query.lower().strip()}|{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - cache_entry['timestamp'] < self._cache_ttl
    
    def _build_classification_prompt(self, query: str, user_context: Dict[str, Any]) -> str:
        """Build optimized classification prompt"""
        user_role = user_context.get("role_name", "Guest")
        
        return f"""You are a precise query classifier for supply chain management systems.

TASK: Classify this query into exactly ONE category:

CATEGORIES:
- document_only: Asks about policies, procedures, definitions, compliance, or guidelines
- database_only: Asks for data, metrics, statistics, lists, counts, or performance analysis  
- hybrid: Requires both policy knowledge AND data analysis
- unclear: Vague, incomplete, or ambiguous queries

USER: {user_role}
QUERY: "{query}"

CRITICAL: Respond with ONLY this exact JSON structure (no extra text):
{{
    "query_type": "document_only",
    "confidence": 0.95,
    "reasoning": "Brief explanation",
    "keywords": ["key", "terms"],
    "suggested_sources": ["documents"]
}}

IMPORTANT: 
- suggested_sources must be an array, use ["documents"] OR ["database"] OR ["documents", "database"]
- Do NOT use "both" - use ["documents", "database"] instead
- Ensure all fields are properly formatted as JSON"""
    
    def classify_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main classification method with caching and comprehensive error handling"""
        start_time = time.time()
        user_context = user_context or {}
        
        # Check cache first
        cache_key = self._generate_cache_key(query, user_context)
        if cache_key in self._classification_cache:
            cache_entry = self._classification_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.info(f"Cache hit for query classification: {query[:50]}...")
                result = cache_entry['result'].copy()
                result['_metrics'] = {
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'confidence_score': result['confidence'],
                    'api_calls_made': 0,
                    'cache_hit': True,
                    'timestamp': time.time()
                }
                return result
        
        try:
            # Primary classification
            classification_result = self._perform_classification(query, user_context)
            
            # Permission analysis
            permissions = self.permission_analyzer.analyze_permissions(query, user_context)
            classification_result['required_permissions'] = permissions
            
            # Cache the result
            self._classification_cache[cache_key] = {
                'result': classification_result.copy(),
                'timestamp': time.time()
            }
            
            # Add metrics
            classification_result['_metrics'] = {
                'processing_time_ms': (time.time() - start_time) * 1000,
                'confidence_score': classification_result['confidence'],
                'api_calls_made': 1,
                'cache_hit': False,
                'timestamp': time.time()
            }
            
            logger.info(f"Query classified: {classification_result['query_type']} "
                       f"(confidence: {classification_result['confidence']:.2f}, "
                       f"permissions: {len(permissions)})")
            
            return classification_result
            
        except Exception as e:
            logger.error(f"Classification failed for query '{query[:50]}...': {e}")
            return self._create_fallback_result(query, str(e), start_time)
    
    def _perform_classification(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual classification using Gemini"""
        prompt = self._build_classification_prompt(query, user_context)
        
        try:
            response = self.model.generate_content(prompt)
            result_text = self._clean_response_text(response.text)
            
            # Parse JSON
            raw_result = json.loads(result_text)
            
            # Fix common format issues before validation
            raw_result = self._normalize_response_format(raw_result)
            
            # Validate using Pydantic model
            validated_result = ClassificationResult(**raw_result)
            
            return validated_result.dict()
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}. Raw response: {response.text[:200]}...")
            raise Exception(f"Invalid JSON response from classification model: {e}")
        except Exception as e:
            logger.error(f"Classification model error: {e}")
            raise

    def _normalize_response_format(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize response format to match expected schema"""
        # Fix suggested_sources if it's a string
        if isinstance(raw_result.get('suggested_sources'), str):
            source_value = raw_result['suggested_sources']
            if source_value == 'both':
                raw_result['suggested_sources'] = ['documents', 'database']
            else:
                raw_result['suggested_sources'] = [source_value]
        
        # Ensure keywords is a list
        if isinstance(raw_result.get('keywords'), str):
            raw_result['keywords'] = [raw_result['keywords']]
        
        # Ensure required_permissions is a list
        if isinstance(raw_result.get('required_permissions'), str):
            raw_result['required_permissions'] = [raw_result['required_permissions']]
        elif raw_result.get('required_permissions') is None:
            raw_result['required_permissions'] = []
        
        return raw_result
    
    def _clean_response_text(self, text: str) -> str:
        """Clean and extract JSON from model response"""
        text = text.strip()
        
        # Remove markdown formatting
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        
        if text.endswith('```'):
            text = text[:-3]
        
        # Extract JSON if embedded in other text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()
        
        return text.strip()
    
    def _create_fallback_result(self, query: str, error_msg: str, start_time: float) -> Dict[str, Any]:
        """Create fallback result when classification fails"""
        # Simple heuristic-based classification
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['policy', 'procedure', 'guideline', 'rule', 'compliance']):
            query_type = "document_only"
            confidence = 0.6
            suggested_sources = ["documents"]
        elif any(word in query_lower for word in ['how many', 'count', 'total', 'list', 'show', 'data']):
            query_type = "database_only" 
            confidence = 0.6
            suggested_sources = ["database"]
        else:
            query_type = "unclear"
            confidence = 0.3
            suggested_sources = ["documents", "database"]
        
        permissions = self.permission_analyzer.analyze_permissions(query)
        
        return {
            "query_type": query_type,
            "confidence": confidence,
            "reasoning": f"Fallback classification due to error: {error_msg}",
            "keywords": query.split()[:5],
            "suggested_sources": suggested_sources,
            "required_permissions": permissions,
            "_metrics": {
                'processing_time_ms': (time.time() - start_time) * 1000,
                'confidence_score': confidence,
                'api_calls_made': 0,
                'cache_hit': False,
                'timestamp': time.time()
            },
            "_fallback": True
        }
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification service statistics"""
        return {
            "cache_size": len(self._classification_cache),
            "cache_ttl_seconds": self._cache_ttl,
            "supported_query_types": [qt.value for qt in QueryType],
            "supported_permissions": [pt.value for pt in PermissionType]
        }

    def clear_cache(self) -> int:
        """Clear classification cache and return number of entries cleared"""
        count = len(self._classification_cache)
        self._classification_cache.clear()
        logger.info(f"Cleared {count} entries from classification cache")
        return count

class IntentClassifier:
    """Backwards compatible wrapper for the enhanced classification service"""
    
    def __init__(self):
        self._service = QueryClassificationService()
        logger.info("IntentClassifier initialized with enhanced classification service")
    
    def classify_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main classification method - backwards compatible interface"""
        return self._service.classify_query(query, user_context)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        return self._service.get_classification_stats()
    
    def clear_cache(self) -> int:
        """Clear classification cache"""
        return self._service.clear_cache()

def create_intent_classifier() -> IntentClassifier:
    """Factory function to create IntentClassifier instance"""
    return IntentClassifier()