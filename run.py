import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Run financial advisor chatbot backend')
    args = parser.parse_args()
    
    if not os.path.exists('.env'):
        print("Error: .env file not found. Please create a .env file with your GEMINI_API_KEY.")
        print("Example: GEMINI_API_KEY=your_api_key_here")
        return 1
    
    # Run the server
    print("Starting financial advisor chatbot backend...")
    os.system("python app_advanced.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 