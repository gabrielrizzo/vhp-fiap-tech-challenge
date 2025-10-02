import unittest
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restrictions.ambulance_patient_restriction import AmbulancePatientRestriction


class TestAmbulancePatientRestriction(unittest.TestCase):
    """Testes unitários para a classe AmbulancePatientRestriction"""

    def setUp(self):
        """Configuração inicial para cada teste"""
        self.restriction = AmbulancePatientRestriction()
        self.sample_route = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        self.sample_vehicle_data = {
            'city_patients': {
                '0.0_0.0': 2,
                '1.0_1.0': 3,
                '2.0_2.0': 1
            }
        }
    
    def test_initialization_default_values(self):
        """Testa inicialização com valores padrão"""
        restriction = AmbulancePatientRestriction()
        self.assertEqual(restriction.name, "ambulance_patient_restriction")
        self.assertEqual(restriction.max_patients, 10)
        self.assertTrue(restriction.is_enabled())
    
    def test_initialization_custom_values(self):
        """Testa inicialização com valores customizados"""
        restriction = AmbulancePatientRestriction(max_patients=20)
        self.assertEqual(restriction.max_patients, 20)
    
    def test_calculate_total_patients_with_vehicle_data(self):
        """Testa cálculo de total de pacientes com dados do veículo"""
        # pylint: disable=protected-access
        # Acesso ao método protegido é necessário para testar a lógica interna
        total_patients = self.restriction._calculate_total_patients(
            self.sample_route, 
            self.sample_vehicle_data
        )
        expected_patients = 2 + 3 + 1  # 6
        self.assertEqual(total_patients, expected_patients)
    
    def test_calculate_total_patients_without_vehicle_data(self):
        """Testa cálculo de total de pacientes sem dados do veículo"""
        # pylint: disable=protected-access
        total_patients = self.restriction._calculate_total_patients(self.sample_route)
        expected_patients = len(self.sample_route)  # 3 (1 paciente por cidade por padrão)
        self.assertEqual(total_patients, expected_patients)
    
    def test_calculate_total_patients_empty_route(self):
        """Testa cálculo de total de pacientes com rota vazia"""
        # pylint: disable=protected-access
        total_patients = self.restriction._calculate_total_patients([])
        self.assertEqual(total_patients, 0)
    
    def test_calculate_total_patients_partial_vehicle_data(self):
        """Testa cálculo de total de pacientes com dados parciais do veículo"""
        # pylint: disable=protected-access
        partial_vehicle_data = {
            'city_patients': {
                '0.0_0.0': 2,
                # '1.0_1.0' não está presente, deve usar valor padrão (1)
                '2.0_2.0': 1
            }
        }
        total_patients = self.restriction._calculate_total_patients(
            self.sample_route, 
            partial_vehicle_data
        )
        expected_patients = 2 + 1 + 1  # 4 (valor padrão 1 para cidade do meio)
        self.assertEqual(total_patients, expected_patients)
    
    def test_validate_route_valid_patients(self):
        """Testa validação de rota com número válido de pacientes"""
        # Rota com total de pacientes 6, capacidade máxima 10
        is_valid = self.restriction.validate_route(
            self.sample_route, 
            self.sample_vehicle_data
        )
        self.assertTrue(is_valid)
    
    def test_validate_route_invalid_patients(self):
        """Testa validação de rota com número inválido de pacientes"""
        # Cria restrição com capacidade menor
        restriction = AmbulancePatientRestriction(max_patients=5)
        is_valid = restriction.validate_route(
            self.sample_route, 
            self.sample_vehicle_data
        )
        self.assertFalse(is_valid)
    
    def test_validate_route_exact_capacity(self):
        """Testa validação de rota com número exato de pacientes"""
        # Rota com total de pacientes 10, capacidade máxima 10
        vehicle_data = {
            'city_patients': {
                '0.0_0.0': 3,
                '1.0_1.0': 4,
                '2.0_2.0': 3
            }
        }
        is_valid = self.restriction.validate_route(
            self.sample_route, 
            vehicle_data
        )
        self.assertTrue(is_valid)
    
    def test_calculate_penalty_no_excess(self):
        """Testa cálculo de penalidade sem excesso de pacientes"""
        penalty = self.restriction.calculate_penalty(
            self.sample_route, 
            self.sample_vehicle_data
        )
        self.assertEqual(penalty, 0.0)
    
    def test_calculate_penalty_with_excess(self):
        """Testa cálculo de penalidade com excesso de pacientes"""
        # Cria restrição com capacidade menor
        restriction = AmbulancePatientRestriction(max_patients=5)
        penalty = restriction.calculate_penalty(
            self.sample_route, 
            self.sample_vehicle_data
        )
        # Total de pacientes: 6, capacidade: 5, excesso: 1, penalidade: 1 * 1000 = 1000
        expected_penalty = 1000.0
        self.assertEqual(penalty, expected_penalty)
    
    def test_calculate_penalty_exact_capacity(self):
        """Testa cálculo de penalidade com número exato de pacientes"""
        vehicle_data = {
            'city_patients': {
                '0.0_0.0': 3,
                '1.0_1.0': 4,
                '2.0_2.0': 3
            }
        }
        penalty = self.restriction.calculate_penalty(
            self.sample_route, 
            vehicle_data
        )
        self.assertEqual(penalty, 0.0)
    
    def test_get_capacity_info_valid_route(self):
        """Testa obtenção de informações de capacidade para rota válida"""
        info = self.restriction.get_capacity_info(
            self.sample_route, 
            self.sample_vehicle_data
        )
        
        expected_info = {
            'total_patients': 6,
            'max_patients': 10,
            'remaining_capacity': 4,
            'capacity_utilization': 0.6,
            'cities_count': 3
        }
        
        self.assertEqual(info, expected_info)
    
    def test_get_capacity_info_invalid_route(self):
        """Testa obtenção de informações de capacidade para rota inválida"""
        restriction = AmbulancePatientRestriction(max_patients=5)
        info = restriction.get_capacity_info(
            self.sample_route, 
            self.sample_vehicle_data
        )
        
        expected_info = {
            'total_patients': 6,
            'max_patients': 5,
            'remaining_capacity': 0,  # Não pode ser negativo
            'capacity_utilization': 1.2,
            'cities_count': 3
        }
        
        self.assertEqual(info, expected_info)
    
    def test_get_capacity_info_empty_route(self):
        """Testa obtenção de informações de capacidade para rota vazia"""
        info = self.restriction.get_capacity_info([])
        
        expected_info = {
            'total_patients': 0,
            'max_patients': 10,
            'remaining_capacity': 10,
            'capacity_utilization': 0.0,
            'cities_count': 0
        }
        
        self.assertEqual(info, expected_info)
    
    def test_get_capacity_info_zero_capacity(self):
        """Testa obtenção de informações com capacidade máxima zero"""
        restriction = AmbulancePatientRestriction(max_patients=0)
        info = restriction.get_capacity_info(self.sample_route)
        
        # Com capacidade zero, utilization deve ser 0 para evitar divisão por zero
        self.assertEqual(info['capacity_utilization'], 0)
        self.assertEqual(info['max_patients'], 0)
    
    def test_inheritance_from_base_restriction(self):
        """Testa se herda corretamente de BaseRestriction"""
        # Testa métodos herdados
        self.assertTrue(hasattr(self.restriction, 'name'))
        self.assertTrue(hasattr(self.restriction, 'is_enabled'))
        self.assertTrue(hasattr(self.restriction, 'enable'))
        self.assertTrue(hasattr(self.restriction, 'disable'))
        self.assertTrue(hasattr(self.restriction, 'get_weight'))
        self.assertTrue(hasattr(self.restriction, 'set_weight'))
        
        # Testa funcionalidade dos métodos herdados
        self.assertTrue(self.restriction.is_enabled())
        
        self.restriction.disable()
        self.assertFalse(self.restriction.is_enabled())
        
        self.restriction.enable()
        self.assertTrue(self.restriction.is_enabled())
        
        self.assertEqual(self.restriction.get_weight(), 1.0)
        self.restriction.set_weight(2.0)
        self.assertEqual(self.restriction.get_weight(), 2.0)
    
    def test_edge_cases_negative_patients(self):
        """Testa casos extremos com número negativo de pacientes"""
        # pylint: disable=protected-access
        vehicle_data = {
            'city_patients': {
                '0.0_0.0': -1,
                '1.0_1.0': 2,
                '2.0_2.0': 3
            }
        }
        
        total_patients = self.restriction._calculate_total_patients(
            self.sample_route, 
            vehicle_data
        )
        expected_patients = -1 + 2 + 3  # 4
        self.assertEqual(total_patients, expected_patients)
        
        # Deve ser válida pois 4 <= 10
        is_valid = self.restriction.validate_route(
            self.sample_route, 
            vehicle_data
        )
        self.assertTrue(is_valid)
    
    def test_edge_cases_very_large_patients(self):
        """Testa casos extremos com número muito grande de pacientes"""
        # pylint: disable=protected-access
        vehicle_data = {
            'city_patients': {
                '0.0_0.0': 1000,
                '1.0_1.0': 2000,
                '2.0_2.0': 3000
            }
        }
        
        total_patients = self.restriction._calculate_total_patients(
            self.sample_route, 
            vehicle_data
        )
        expected_patients = 6000
        self.assertEqual(total_patients, expected_patients)
        
        # Deve ser inválida pois 6000 > 10
        is_valid = self.restriction.validate_route(
            self.sample_route, 
            vehicle_data
        )
        self.assertFalse(is_valid)
        
        # Penalidade deve ser alta
        penalty = self.restriction.calculate_penalty(
            self.sample_route, 
            vehicle_data
        )
        expected_penalty = (6000 - 10) * 1000  # 5990000
        self.assertEqual(penalty, expected_penalty)
    
    def test_route_with_duplicate_cities(self):
        """Testa rota com cidades duplicadas"""
        # pylint: disable=protected-access
        route_with_duplicates = [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
        vehicle_data = {
            'city_patients': {
                '0.0_0.0': 2,
                '1.0_1.0': 3
            }
        }
        
        total_patients = self.restriction._calculate_total_patients(
            route_with_duplicates, 
            vehicle_data
        )
        # Deve somar o número de pacientes da cidade duplicada também
        expected_patients = 2 + 3 + 2  # 7
        self.assertEqual(total_patients, expected_patients)
    
    def test_vehicle_data_without_city_patients(self):
        """Testa dados do veículo sem city_patients"""
        # pylint: disable=protected-access
        vehicle_data = {'other_data': 'value'}
        total_patients = self.restriction._calculate_total_patients(
            self.sample_route, 
            vehicle_data
        )
        # Deve usar número padrão de pacientes por cidade (1)
        expected_patients = len(self.sample_route)  # 3
        self.assertEqual(total_patients, expected_patients)
    
    def test_vehicle_data_none(self):
        """Testa com vehicle_data None"""
        # pylint: disable=protected-access
        total_patients = self.restriction._calculate_total_patients(
            self.sample_route, 
            None
        )
        # Deve usar número padrão de pacientes por cidade (1)
        expected_patients = len(self.sample_route)  # 3
        self.assertEqual(total_patients, expected_patients)


if __name__ == '__main__':
    # Configuração para executar os testes
    unittest.main(verbosity=2)