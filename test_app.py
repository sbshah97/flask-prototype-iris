#!/usr/bin/env python3
"""
Simple test script to verify the Flask app and RAG pipeline work correctly.
Run this after setting up your environment to test the application.
"""

import os
import sys
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    try:
        import app
        import rag_pipeline
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("🔍 Testing environment configuration...")
    
    # Check for required API key
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("⚠️  GOOGLE_API_KEY not found - you'll need this for Gemini")
        return False
    
    print("✅ Environment configuration looks good")
    return True

def test_flask_app(port=8001):
    """Test Flask app endpoints"""
    print(f"🔍 Testing Flask app on port {port}...")
    
    base_url = f"http://localhost:{port}"
    
    # Start the app in the background (you'll need to do this manually)
    print(f"Please start the app manually with: python app.py")
    print("Press Enter when the app is running...")
    input()
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test status endpoint
        response = requests.get(f"{base_url}/status", timeout=10)
        if response.status_code == 200:
            print("✅ Status endpoint working")
            print(f"Status: {response.json()}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
        
        # Test chat endpoint with a simple question
        test_question = "What are the attendance requirements?"
        response = requests.post(
            f"{base_url}/chat",
            json={"question": test_question},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat endpoint working")
            print(f"Question: {test_question}")
            print(f"Answer: {result.get('answer', '')[:200]}...")
            print(f"Sources found: {len(result.get('sources', []))}")
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        print("Make sure the Flask app is running on the correct port")
        return False

def test_rag_pipeline():
    """Test RAG pipeline directly"""
    print("🔍 Testing RAG pipeline directly...")
    
    try:
        from rag_pipeline import get_pipeline, initialize_pipeline
        
        # Initialize pipeline
        print("Initializing RAG pipeline...")
        success = initialize_pipeline(force_rebuild=False)
        
        if not success:
            print("❌ RAG pipeline initialization failed")
            return False
        
        # Test query
        pipeline = get_pipeline()
        result = pipeline.query("What is NITK?")
        
        if result["success"]:
            print("✅ RAG pipeline working")
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Sources: {len(result['sources'])}")
        else:
            print(f"❌ RAG query failed: {result['answer']}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ RAG pipeline error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 NITK Academic Advisor - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("RAG Pipeline Test", test_rag_pipeline),
        ("Flask App Test", lambda: test_flask_app())
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your NITK Academic Advisor is ready!")
    else:
        print("🔧 Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()