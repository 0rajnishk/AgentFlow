# app/api/v1/auth/models.py

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime # Import datetime for default timestamp
from app.database import Base

class Role(Base):
    __tablename__ = "Roles"
    RoleId = Column(Integer, primary_key=True, autoincrement=True)
    RoleName = Column(String, unique=True, nullable=False)

    users = relationship("User", back_populates="role")

class User(Base):
    __tablename__ = "Users"
    UserId = Column(Integer, primary_key=True, autoincrement=True)
    Username = Column(String, unique=True, nullable=False)
    Email = Column(String)
    PasswordHash = Column(String, nullable=False)
    RoleId = Column(Integer, ForeignKey("Roles.RoleId"))
    Region = Column(String)

    role = relationship("Role", back_populates="users")
    # Make sure 'Chat' model is defined before or in the same file
    chats = relationship("Chat", back_populates="user")

# Add the Chat and Message models here
class Chat(Base):
    __tablename__ = "Chats"
    ChatId = Column(Integer, primary_key=True, autoincrement=True)
    UserId = Column(Integer, ForeignKey("Users.UserId"))
    ChatTitle = Column(String)
    CreatedAt = Column(DateTime, default=datetime.utcnow) # Use DateTime type for timestamps

    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat")

class Message(Base):
    __tablename__ = "Messages"
    MessageId = Column(Integer, primary_key=True, autoincrement=True)
    ChatId = Column(Integer, ForeignKey("Chats.ChatId"))
    Role = Column(String) # 'user' or 'assistant'
    Content = Column(Text, nullable=False) # Use Text for potentially long content
    Timestamp = Column(DateTime, default=datetime.utcnow) # Use DateTime type

    chat = relationship("Chat", back_populates="messages")

# Optional: Add other models if you want SQLAlchemy to manage them
# This is assuming you might eventually want ORM for Customers, Products, Orders too
class Customer(Base):
    __tablename__ = "Customers"
    CustomerId = Column(Integer, primary_key=True)
    Fname = Column(String)
    Lname = Column(String)
    Email = Column(String)
    City = Column(String)
    State = Column(String)
    Country = Column(String)
    Zipcode = Column(String)
    Segment = Column(String)
    Street = Column(String)
    Password = Column(String) # Consider hashing this if customers log in via this table

    orders = relationship("Order", back_populates="customer")

class Product(Base):
    __tablename__ = "Products"
    ProductCardId = Column(Integer, primary_key=True)
    ProductCategoryId = Column(Integer)
    ProductName = Column(String)
    ProductDescription = Column(String)
    ProductPrice = Column(Text) # Use Text for REAL in SQLite or Numeric/Float
    ProductImage = Column(String)
    ProductStatus = Column(Integer)

    orders = relationship("Order", back_populates="product")

class Order(Base):
    __tablename__ = "Orders"
    OrderId = Column(Integer, primary_key=True)
    OrderDate = Column(DateTime)
    ShippingDate = Column(DateTime)
    OrderStatus = Column(String)
    OrderRegion = Column(String)
    OrderCountry = Column(String)
    OrderState = Column(String)
    OrderCity = Column(String)
    OrderZipcode = Column(String)
    OrderCustomerId = Column(Integer, ForeignKey("Customers.CustomerId"))
    ProductCardId = Column(Integer, ForeignKey("Products.ProductCardId"))
    SalesPerCustomer = Column(Text) # Use Text for REAL in SQLite or Numeric/Float
    OrderProfitPerOrder = Column(Text) # Use Text for REAL in SQLite or Numeric/Float
    OrderItemTotal = Column(Text) # Use Text for REAL in SQLite or Numeric/Float
    ShippingMode = Column(String)
    DeliveryStatus = Column(String)
    LateDeliveryRisk = Column(Integer)

    customer = relationship("Customer", back_populates="orders")
    product = relationship("Product", back_populates="orders")