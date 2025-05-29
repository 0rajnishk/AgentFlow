# app/api/v1/auth/auth.py

from builtins import int, str
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer # Keep this if you still use it for other routes
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr # Make sure BaseModel is imported
from datetime import timedelta
from typing import Optional

from app.database import get_db
from app.api.v1.auth.models import User, Role
from app.utils.security import verify_password, get_password_hash, create_access_token
from app.core.config import get_settings
from app.core.logging import logger
from app.utils.security import verify_token

router = APIRouter()

# Get the settings instance by calling the function
settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token") # Keep this if other routes use it for header parsing

# Pydantic models (your existing ones)
class UserCreate(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    password: str
    region: Optional[str] = None
    role_name: Optional[str] = "User"

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: Optional[EmailStr] = None
    role_name: Optional[str] = None
    region: Optional[str] = None

    class Config:
        from_attributes = True

# class Token(BaseModel):
#     access_token: str
#     token_type: str
#     user_id: int
#     username: str
#     email: Optional[EmailStr]
#     role_name: Optional[str]
#     region: Optional[str]

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None
    role_name: Optional[str] = None

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    username: str
    email: Optional[EmailStr]
    role_name: Optional[str]
    region: Optional[str]


# New Pydantic model for login request
class LoginRequest(BaseModel):
    username: str
    password: str

# Dependency to get current user (remains unchanged as it uses OAuth2PasswordBearer for header parsing)
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = verify_token(token)
    if payload is None:
        raise credentials_exception
    username: str = payload.get("sub")
    user_id: int = payload.get("user_id")
    if username is None or user_id is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.UserId == user_id).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    return current_user

# --- Routes ---

@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.Username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    role = db.query(Role).filter(Role.RoleName == user.role_name).first()
    if not role:
        new_role = Role(RoleName=user.role_name)
        db.add(new_role)
        db.commit()
        db.refresh(new_role)
        role = new_role

    hashed_password = get_password_hash(user.password)
    new_user = User(
        Username=user.username,
        Email=user.email,
        PasswordHash=hashed_password,
        RoleId=role.RoleId,
        Region=user.region
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    new_user.role = role
    
    logger.info(f"User {new_user.Username} signed up successfully.")
    return UserResponse(
        user_id=new_user.UserId,
        username=new_user.Username,
        email=new_user.Email,
        role_name=role.RoleName,
        region=new_user.Region
    )

# Modified /token endpoint
@router.post("/token", response_model=LoginResponse)
async def login_for_access_token(request: LoginRequest, db: Session = Depends(get_db)): # Changed from form_data: OAuth2PasswordRequestForm
    user = db.query(User).filter(User.Username == request.username).first() # Use request.username
    if not user or not verify_password(request.password, user.PasswordHash): # Use request.password
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    role = db.query(Role).filter(Role.RoleId == user.RoleId).first()
    role_name = role.RoleName if role else "Unknown"

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.Username, "user_id": user.UserId, "role_name": role_name},
        expires_delta=access_token_expires
    )
    logger.info(f"User {user.Username} logged in successfully.")
    # return {"access_token": access_token, "token_type": "bearer", "user_id": user.UserId, "username": user.Username, "email": user.Email, "role_name": role_name, "region": user.Region}
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.UserId,
        "username": user.Username,
        "email": user.Email,
        "role_name": role_name,
        "region": user.Region
    }




@router.get("/users/me/", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return UserResponse(
        user_id=current_user.UserId,
        username=current_user.Username,
        email=current_user.Email,
        role_name=current_user.role.RoleName if current_user.role else "Unknown",
        region=current_user.Region
    )
