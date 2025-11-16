"""
Configuration Manager for loading YAML configs
"""

import yaml
import os
from typing import Dict, Any
from argparse import Namespace


class ConfigManager:
    """Manage experiment configurations from YAML files"""

    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to YAML config file
            config_dict: Configuration dictionary (overrides config_path)
        """
        if config_dict:
            self.config = config_dict
        elif config_path:
            self.config = self.load_yaml(config_path)
        else:
            self.config = {}

    @staticmethod
    def load_yaml(config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default

        return value if value is not None else default

    def update(self, updates: Dict):
        """Update configuration with dictionary"""
        self._deep_update(self.config, updates)

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def to_namespace(self) -> Namespace:
        """Convert config to argparse Namespace"""
        flat_config = self._flatten_dict(self.config)
        return Namespace(**flat_config)

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def save(self, save_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def __str__(self) -> str:
        """String representation"""
        return yaml.dump(self.config, default_flow_style=False, indent=2)

    def __repr__(self) -> str:
        return f"ConfigManager({self.config})"


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configurations, with override_config taking precedence

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    ConfigManager(config_dict=merged).update(override_config)
    return merged
