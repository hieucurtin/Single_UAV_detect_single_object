import yaml
import os

def load_config(config_name):
    """
    Load configuration from YAML file.
    
    Args:
        config_name: Name of the config file (without .yaml extension)
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', f'{config_name}.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
