#!/usr/bin/env python3
"""
Database table setup script for MySQL
Assumes the database already exists and only creates tables
"""

import os
import sys
import traceback

# Add explicit error handling for imports
try:
    print("Importing database module...")
    from database import init_database
    print("Successfully imported database module")
except Exception as import_error:
    print(f"❌ Error importing database module: {import_error}")
    traceback.print_exc()
    sys.exit(1)

def setup_tables():
    """Setup database tables (assumes database already exists)"""
    try:
        print("🗄️  Setting up database tables...")
        print("=" * 60)
        
        # Check environment variables
        print("Database Configuration:")
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '3306')
        user = os.getenv('DB_USER', 'root')
        password = os.getenv('DB_PASSWORD', 'Cruise#7788!') # Default from database.py
        db_name = os.getenv('DB_NAME', 'forecasting_db')
        
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"User: {user}")
        print(f"Password: {'*' * len(password) if password else 'Not set'}")
        print(f"Database: {db_name}")
        print()
        
        print("📋 Note: This script will now attempt to create the database if it doesn't exist.")
        print("📋 It will also create the required tables if they don't exist.")
        print()
        
        # Test MySQL connection directly before initializing tables
        print("Testing direct MySQL connection...")
        try:
            import mysql.connector
            
            # First try connecting without specifying the database
            print("Checking MySQL server connection...")
            root_conn = mysql.connector.connect(
                host=host,
                port=int(port),
                user=user,
                password=password
            )
            print("✅ MySQL server connection successful!")
            
            # Check if database exists
            cursor = root_conn.cursor()
            cursor.execute(f"SHOW DATABASES LIKE '{db_name}'")
            result = cursor.fetchone()
            
            if not result:
                print(f"⚠️ Database '{db_name}' does not exist. Attempting to create it...")
                cursor.execute(f"CREATE DATABASE {db_name}")
                print(f"✅ Database '{db_name}' created successfully!")
            else:
                print(f"✅ Database '{db_name}' already exists.")
                
            cursor.close()
            root_conn.close()
            
            # Now connect to the specific database
            conn = mysql.connector.connect(
                host=host,
                port=int(port),
                user=user,
                password=password,
                database=db_name
            )
            print("✅ Direct MySQL connection to database successful!")
            conn.close()
        except Exception as mysql_err:
            print(f"❌ Direct MySQL connection test failed: {mysql_err}")
            print("This indicates a problem with MySQL connection, not with SQLAlchemy.")
            traceback.print_exc()
            
        print("\nAttempting to initialize database tables...")
        # Initialize tables
        if init_database():
            print("\n🎉 Database tables setup completed!")
            print("\nTables created/verified:")
            print("- forecast_data (stores uploaded forecast data)")
            print("- external_factor_data (stores external factor data)")
            print("- forecast_configurations (stores saved configurations)")
            print("\nYou can now start the application with: python main.py")
            return True
        else:
            print("\n❌ Database setup failed. Please check the error messages above.")
            print("\nTroubleshooting:")
            print("1. Ensure MySQL server is running")
            print("2. Verify the database 'forecasting_db' exists")
            print("3. Check database credentials in .env file")
            print("4. Ensure user has CREATE TABLE privileges")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\nDetailed error information:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Script path: {os.path.abspath(__file__)}")
    
    # Check for required packages
    try:
        import mysql.connector
        print("✅ mysql-connector-python is installed")
    except ImportError:
        print("❌ mysql-connector-python is not installed")
        print("Please install it with: pip install mysql-connector-python")
        sys.exit(1)
        
    try:
        import sqlalchemy
        print(f"✅ SQLAlchemy is installed (version: {sqlalchemy.__version__})")
    except ImportError:
        print("❌ SQLAlchemy is not installed")
        print("Please install it with: pip install sqlalchemy")
        sys.exit(1)
    
    # Run the setup
    result = setup_tables()
    
    # Print final status
    if result:
        print("\n✅ Database setup completed successfully!")
    else:
        print("\n❌ Database setup failed. Please check the error messages above.")
        
    print("\nIf you're still having issues, try:")
    print("1. Verify MySQL is running with: mysql --version")
    print("2. Check if you can connect manually: mysql -u root -p")
    print("3. Ensure all required packages are installed: pip install -r requirements.txt")
    print("4. Check if the user has CREATE privileges in MySQL")