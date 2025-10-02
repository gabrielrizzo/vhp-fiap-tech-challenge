import json
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    def __init__(self, config_file: str = "config/medical_tsp_config.json"):
        self.config_file = Path(config_file)
        self.default_config = {
            "display": {
                "width": 1500,
                "height": 800,
                "fps": 30,
                "node_radius": 10,
                "plot_x_offset": 450
            },
            "genetic_algorithm": {
                "population_size": 2000,
                "generation_limit": 10000,
                "n_cities": 48,
                "n_exploration_generation": 1000,
                "n_neighbors": 500
            },
            "mutation": {
                "initial_probability": 0.85,
                "initial_intensity": 25,
                "after_exploration_intensity": 5,
                "after_exploration_probability": 0.5
            },
            "diversity": {
                "threshold": 0.3,
                "injection_rate": 0.1,
                "monitor_frequency": 50
            },
            "restrictions": {
                "fuel": {
                    "enabled": True,
                    "max_distance": 250.0,
                    "fuel_cost_per_km": 0.8,
                    "weight": 1.0
                },
                "vehicle_capacity": {
                    "enabled": True,
                    "max_patients": 10,
                    "delivery_weight_per_city": 1.0,
                    "weight": 1.0
                }
            },
            "llm": {
                "enabled": True,
                "model": "gpt-3.5-turbo",
                "max_tokens": 1500,
                "temperature": 0.7,
                "fallback_mode": True
            },
            "logging": {
                "level": "INFO",
                "log_file": "medical_tsp.log",
                "log_generations": True,
                "log_frequency": 100
            }
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                return self._merge_configs(self.default_config, loaded_config)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Error loading config file. Using defaults.")
                return self.default_config.copy()
        else:
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any] = None):
        config_to_save = config if config is not None else self.config
        with open(self.config_file, 'w') as f:
            json.dump(config_to_save, f, indent=4)
    
    def get(self, key_path: str, default=None):
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any):
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
            
        config_ref[keys[-1]] = value
    
    def update_restriction_config(self, restriction_name: str, settings: Dict[str, Any]):
        restriction_path = f"restrictions.{restriction_name}"
        current_settings = self.get(restriction_path, {})
        current_settings.update(settings)
        self.set(restriction_path, current_settings)
    
    def is_restriction_enabled(self, restriction_name: str) -> bool:
        return self.get(f"restrictions.{restriction_name}.enabled", False)
    
    def get_restriction_weight(self, restriction_name: str) -> float:
        return self.get(f"restrictions.{restriction_name}.weight", 1.0)
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        merged = default.copy()
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def print_config(self):
        print("Current Configuration:")
        print("-" * 30)
        self._print_config_section(self.config)
    
    def _print_config_section(self, config: Dict[str, Any], indent: int = 0):
        for key, value in config.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_config_section(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
