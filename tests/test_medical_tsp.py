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
        from restrictions.route_cost_restriction import RouteCostRestriction
        from restrictions.multiple_vehicles import MultipleVehiclesRestriction
        from restrictions.fixed_start_restriction import FixedStartRestriction
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    test_route = [(0, 0), (10, 0), (10, 10), (0, 10)]

    from restrictions.fuel_restriction import FuelRestriction
    fuel_restriction = FuelRestriction(max_distance=100.0)
    is_valid = fuel_restriction.validate_route(test_route)
    penalty = fuel_restriction.calculate_penalty(test_route)
    print(f"✓ Basic test - Fuel Restriction - Valid: {is_valid}, Penalty: {penalty}")

    from restrictions.vehicle_capacity_restriction import VehicleCapacityRestriction
    vehicle_capacity_restriction = VehicleCapacityRestriction(max_capacity=10)
    is_valid = vehicle_capacity_restriction.validate_route(test_route)
    penalty = vehicle_capacity_restriction.calculate_penalty(test_route)
    print(f"✓ Basic test - Vehicle Capacity Restriction - Valid: {is_valid}, Penalty: {penalty}")

    from restrictions.fixed_start_restriction import FixedStartRestriction
    fixed_start_restriction = FixedStartRestriction(hospital_location=(0, 0))
    is_valid = fixed_start_restriction.validate_route(test_route)
    penalty = fixed_start_restriction.calculate_penalty(test_route)
    print(f"✓ Basic test - Fixed Start Restriction - Valid: {is_valid}, Penalty: {penalty}")

    from restrictions.route_cost_restriction import RouteCostRestriction
    route_cost_restriction = RouteCostRestriction(cities_locations=test_route, route_cost_dict={})
    is_valid = route_cost_restriction.validate_route(test_route)
    penalty = route_cost_restriction.calculate_penalty(test_route)
    print(f"✓ Basic test - Route Cost Restriction - Valid: {is_valid}, Penalty: {penalty}")

    from restrictions.multiple_vehicles import MultipleVehiclesRestriction
    multiple_vehicles_restriction = MultipleVehiclesRestriction(max_vehicles=10, vehicle_capacity=10)
    is_valid = multiple_vehicles_restriction.validate_route(test_route)
    penalty = multiple_vehicles_restriction.calculate_penalty(test_route)
    print(f"✓ Basic test - Multiple Vehicles Restriction - Valid: {is_valid}, Penalty: {penalty}")

    return True

if __name__ == "__main__":
    print("Running Medical TSP tests...")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
