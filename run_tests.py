import sys
import traceback

def run_test_module(module_name, module_path):
    print(f"Running tests in {module_name}...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Find all functions starting with test_
    test_funcs = [getattr(module, name) for name in dir(module) if name.startswith("test_") and callable(getattr(module, name))]
    
    passed = 0
    failed = 0
    for func in test_funcs:
        print(f"  - {func.__name__}...", end="", flush=True)
        try:
            func()
            print(" PASSED")
            passed += 1
        except Exception as e:
            print(" FAILED")
            traceback.print_exc()
            failed += 1
            
    return passed, failed

def main():
    print("=" * 50)
    print("RUNNING NANOFORECAST UNIT TESTS (SELF-CONTAINED RUNNER)")
    print("=" * 50)
    
    passed_total = 0
    failed_total = 0
    
    # Run model tests
    p, f = run_test_module("test_model", "tests/test_model.py")
    passed_total += p
    failed_total += f
    
    # Run data tests
    p, f = run_test_module("test_data", "tests/test_data.py")
    passed_total += p
    failed_total += f
    
    print("=" * 50)
    print(f"Test Summary: {passed_total} passed, {failed_total} failed")
    print("=" * 50)
    
    if failed_total > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
