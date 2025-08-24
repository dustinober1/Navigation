#!/usr/bin/env python3
"""
Test runner for the Navigation DQN project.
"""

import unittest
import sys
import os
import argparse
from io import StringIO


def discover_and_run_tests(test_dir='tests', pattern='test_*.py', verbosity=2):
    """Discover and run all tests in the specified directory."""
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(current_dir, test_dir)
    
    if not os.path.exists(start_dir):
        print(f"Test directory '{start_dir}' not found!")
        return False
    
    suite = loader.discover(start_dir, pattern=pattern)
    
    # Count total tests
    total_tests = count_tests(suite)
    print(f"Discovered {total_tests} tests\n")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print_test_summary(result, total_tests)
    
    return result.wasSuccessful()


def count_tests(suite):
    """Count the total number of tests in a test suite."""
    count = 0
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            count += count_tests(test)
        else:
            count += 1
    return count


def print_test_summary(result, total_tests):
    """Print a summary of test results."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print("\n❌ SOME TESTS FAILED")
        
        if result.failures:
            print(f"\nFailures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}")
                
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    print("="*60)


def run_specific_test(test_name, verbosity=2):
    """Run a specific test by name."""
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Load and run specific test
    loader = unittest.TestLoader()
    
    try:
        suite = loader.loadTestsFromName(test_name)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result.wasSuccessful()
    except Exception as e:
        print(f"Error loading test '{test_name}': {e}")
        return False


def list_available_tests():
    """List all available test modules and classes."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(current_dir, 'tests')
    
    if not os.path.exists(test_dir):
        print("No tests directory found!")
        return
    
    print("Available test modules:")
    print("-" * 30)
    
    for filename in os.listdir(test_dir):
        if filename.startswith('test_') and filename.endswith('.py'):
            module_name = filename[:-3]  # Remove .py extension
            print(f"  {module_name}")
            
            # Try to load the module and list test classes
            try:
                sys.path.insert(0, current_dir)
                module = __import__(f'tests.{module_name}', fromlist=[''])
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, unittest.TestCase) and 
                        attr != unittest.TestCase):
                        print(f"    └── {attr_name}")
                        
                        # List test methods
                        for method_name in dir(attr):
                            if method_name.startswith('test_'):
                                print(f"        └── {method_name}")
                                
            except Exception as e:
                print(f"    └── Error loading module: {e}")
            
            print()


def main():
    parser = argparse.ArgumentParser(description='Run tests for Navigation DQN project')
    parser.add_argument('--test', type=str, help='Run a specific test (e.g., tests.test_models.TestQNetwork.test_forward_pass)')
    parser.add_argument('--list', action='store_true', help='List all available tests')
    parser.add_argument('--pattern', type=str, default='test_*.py', help='Test file pattern (default: test_*.py)')
    parser.add_argument('--verbosity', '-v', type=int, default=2, choices=[0, 1, 2], 
                        help='Test output verbosity (0=quiet, 1=normal, 2=verbose)')
    parser.add_argument('--directory', '-d', type=str, default='tests', help='Test directory (default: tests)')
    
    args = parser.parse_args()
    
    print("Navigation DQN Project - Test Runner")
    print("="*40)
    
    if args.list:
        list_available_tests()
        return
        
    if args.test:
        print(f"Running specific test: {args.test}")
        success = run_specific_test(args.test, args.verbosity)
    else:
        print("Running all tests...")
        success = discover_and_run_tests(args.directory, args.pattern, args.verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()