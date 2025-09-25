#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    try:
        from core.base_restriction import BaseRestriction
        from core.restriction_manager import RestrictionManager
        from restrictions.fuel_restriction import FuelRestriction
        from restrictions.vehicle_capacity_restriction import VehicleCapacityRestriction
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    from restrictions.fuel_restriction import FuelRestriction
    
    fuel_restriction = FuelRestriction(max_distance=100.0)
    test_route = [(0, 0), (10, 0), (10, 10), (0, 10)]
    
    is_valid = fuel_restriction.validate_route(test_route)
    penalty = fuel_restriction.calculate_penalty(test_route)
    
    print(f"✓ Basic test - Valid: {is_valid}, Penalty: {penalty}")
    return True

if __name__ == "__main__":
    print("Running basic tests...")
    if test_imports() and test_basic_functionality():
        print("All tests passed!")
    else:
        print("Some tests failed!")
