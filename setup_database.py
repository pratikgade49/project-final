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

def test_auth_methods(host, port, user, password):
    """Test different authentication methods for MySQL"""
    import mysql.connector
    
    print("\nTesting different authentication methods...")
    
    # Method 1: Standard authentication
    try:
        print("Trying standard authentication...")
        conn = mysql.connector.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            connection_timeout=5
        )
        print("✅ Standard authentication successful!")
        conn.close()
        return True
    except mysql.connector.Error as err:
        print(f"❌ Standard authentication failed: {err}")
    
    # Method 2: Try with auth_plugin
    try:
        print("Trying mysql_native_password authentication...")
        conn = mysql.connector.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            auth_plugin='mysql_native_password',
            connection_timeout=5
        )
        print("✅ mysql_native_password authentication successful!")
        conn.close()
        return True
    except mysql.connector.Error as err:
        print(f"❌ mysql_native_password authentication failed: {err}")
    
    # Method 3: Try with caching_sha2_password
    try:
        print("Trying caching_sha2_password authentication...")
        conn = mysql.connector.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            auth_plugin='caching_sha2_password',
            connection_timeout=5
        )
        print("✅ caching_sha2_password authentication successful!")
        conn.close()
        return True
    except mysql.connector.Error as err:
        print(f"❌ caching_sha2_password authentication failed: {err}")
    
    print("\n❌ All authentication methods failed.")
    print("This suggests an issue with the MySQL user credentials or authentication setup.")
    return False

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
            
            # Test if we can connect at all with any authentication method
            auth_success = test_auth_methods(host, port, user, password)
            
            if not auth_success:
                print("\n❌ Could not authenticate with MySQL using any method.")
                print("Please check your MySQL user and password.")
                print("You may need to:")
                print("1. Reset the MySQL root password")
                print("2. Create a new MySQL user with appropriate permissions")
                print("3. Set the correct credentials in environment variables:")
                print("   set DB_USER=your_mysql_user")
                print("   set DB_PASSWORD=your_mysql_password")
                
                # Ask if they want to try a different password
                try_different = input("\nWould you like to try a different password? (y/n): ")
                if try_different.lower() == 'y':
                    new_password = input("Enter MySQL password for user 'root': ")
                    if new_password:
                        password = new_password
                        print("Trying with the new password...")
                        auth_success = test_auth_methods(host, port, user, password)
                        if not auth_success:
                            print("❌ Still unable to authenticate with the new password.")
                            raise Exception("Authentication failed with all methods")
                else:
                    raise Exception("Authentication failed with all methods")
            
            # Now try to connect to MySQL server
            try:
                print(f"\nAttempting to connect to MySQL at {host}:{port} with user '{user}'...")
                root_conn = mysql.connector.connect(
                    host=host,
                    port=int(port),
                    user=user,
                    password=password,
                    connection_timeout=10  # Add timeout to avoid hanging
                )
                print("✅ MySQL server connection successful!")
            except mysql.connector.Error as err:
                print(f"❌ MySQL Connection Error: {err}")
                if err.errno == 1045:  # Access denied error
                    print("This is likely a password or user authentication issue.")
                    print("Please check that the MySQL user and password are correct.")
                elif err.errno == 2003:  # Can't connect to server
                    print("Cannot connect to MySQL server. Please check if:")
                    print("1. MySQL server is running")
                    print("2. The host and port are correct")
                    print("3. Firewall is not blocking the connection")
                else:
                    print(f"MySQL Error Code: {err.errno}")
                raise
            
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

def check_mysql_running():
    """Check if MySQL server is running by attempting to execute the mysql command"""
    import subprocess
    import platform
    
    print("Checking if MySQL server is running...")
    
    try:
        if platform.system() == "Windows":
            # On Windows, check services
            result = subprocess.run(
                ["sc", "query", "mysql"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if "RUNNING" in result.stdout:
                print("✅ MySQL service appears to be running (Windows service check)")
                return True
            else:
                print("⚠️ MySQL service does not appear to be running (Windows service check)")
                # Try alternative check
                try:
                    result = subprocess.run(
                        ["mysql", "--version"], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    print(f"MySQL client found: {result.stdout.strip()}")
                    # Just because client exists doesn't mean server is running
                    # Try a quick connection
                    import socket
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    result = s.connect_ex(('localhost', 3306))
                    s.close()
                    if result == 0:
                        print("✅ MySQL server appears to be running (port 3306 is open)")
                        return True
                    else:
                        print("❌ MySQL server does not appear to be running (port 3306 is closed)")
                        return False
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("❌ MySQL client not found in PATH")
                    return False
        else:
            # On Linux/Mac, use different command
            result = subprocess.run(
                ["pgrep", "mysqld"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.stdout.strip():
                print("✅ MySQL server appears to be running (process found)")
                return True
            else:
                print("⚠️ MySQL server process not found")
                # Try service status as fallback
                try:
                    result = subprocess.run(
                        ["systemctl", "status", "mysql"], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    if "active (running)" in result.stdout:
                        print("✅ MySQL service is active and running")
                        return True
                    else:
                        print("❌ MySQL service is not running")
                        return False
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("⚠️ Could not check MySQL service status")
                    return False
    except Exception as e:
        print(f"⚠️ Error checking MySQL status: {e}")
        return False

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Script path: {os.path.abspath(__file__)}")
    
    # Check for required packages
    try:
        import mysql.connector
        print(f"✅ mysql-connector-python is installed (version: {mysql.connector.__version__})")
    except ImportError:
        print("❌ mysql-connector-python is not installed")
        print("Please install it with: pip install mysql-connector-python")
        sys.exit(1)
        
    # Check if MySQL is running
    mysql_running = check_mysql_running()
    
    if not mysql_running:
        print("\n❌ MySQL server does not appear to be running!")
        print("Please start your MySQL server before continuing.")
        print("On Windows: Open Services and start the MySQL service")
        print("On Linux: sudo systemctl start mysql")
        print("On macOS: sudo brew services start mysql")
        user_input = input("\nDo you want to continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting script. Please start MySQL and try again.")
            sys.exit(1)
        print("Continuing despite MySQL server check failure...")
        
    try:
        import sqlalchemy
        print(f"✅ SQLAlchemy is installed (version: {sqlalchemy.__version__})")
    except ImportError:
        print("❌ SQLAlchemy is not installed")
        print("Please install it with: pip install sqlalchemy")
        sys.exit(1)
    
    # Run the setup
    print("\nAttempting database setup...")
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