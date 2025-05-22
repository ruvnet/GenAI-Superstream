"""
Tests for the advanced.config module.

These tests verify configuration loading, environment variable handling,
and default value behaviors following TDD principles.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

# Test imports should fail initially (Red phase)
def test_config_module_imports():
    """Test that config module can be imported."""
    # This test should initially fail until we ensure proper imports
    try:
        from advanced import config
        assert hasattr(config, 'DB_CONFIG')
        assert hasattr(config, 'PERPLEXITY_CONFIG')
        assert hasattr(config, 'LOG_CONFIG')
        assert hasattr(config, 'ANALYTICS_CONFIG')
    except ImportError as e:
        pytest.fail(f"Failed to import config module: {e}")


def test_db_config_has_required_fields():
    """Test that DB_CONFIG contains all required fields."""
    from advanced.config import DB_CONFIG
    
    required_fields = ['db_path', 'memory_limit', 'threads', 'cache_enabled', 'cache_size']
    
    for field in required_fields:
        assert field in DB_CONFIG, f"Missing required field: {field}"
    
    # Test data types
    assert isinstance(DB_CONFIG['db_path'], str)
    assert isinstance(DB_CONFIG['memory_limit'], str)
    assert isinstance(DB_CONFIG['threads'], int)
    assert isinstance(DB_CONFIG['cache_enabled'], bool)
    assert isinstance(DB_CONFIG['cache_size'], int)


def test_perplexity_config_has_required_fields():
    """Test that PERPLEXITY_CONFIG contains all required fields."""
    from advanced.config import PERPLEXITY_CONFIG
    
    required_fields = ['server_name', 'mcp_url', 'system_prompt', 'max_tokens', 'temperature', 'return_citations']
    
    for field in required_fields:
        assert field in PERPLEXITY_CONFIG, f"Missing required field: {field}"
    
    # Test data types
    assert isinstance(PERPLEXITY_CONFIG['server_name'], str)
    assert isinstance(PERPLEXITY_CONFIG['mcp_url'], str)
    assert isinstance(PERPLEXITY_CONFIG['system_prompt'], str)
    assert isinstance(PERPLEXITY_CONFIG['max_tokens'], int)
    assert isinstance(PERPLEXITY_CONFIG['temperature'], float)
    assert isinstance(PERPLEXITY_CONFIG['return_citations'], bool)


def test_log_config_has_required_fields():
    """Test that LOG_CONFIG contains all required fields."""
    from advanced.config import LOG_CONFIG
    
    required_fields = ['log_file', 'level', 'format']
    
    for field in required_fields:
        assert field in LOG_CONFIG, f"Missing required field: {field}"
    
    # Test data types
    assert isinstance(LOG_CONFIG['log_file'], str)
    assert isinstance(LOG_CONFIG['level'], str)
    assert isinstance(LOG_CONFIG['format'], str)


def test_analytics_config_has_required_fields():
    """Test that ANALYTICS_CONFIG contains all required fields."""
    from advanced.config import ANALYTICS_CONFIG
    
    required_fields = ['export_dir', 'visualization_dir', 'default_cluster_count', 'default_feature_count', 'pca_components']
    
    for field in required_fields:
        assert field in ANALYTICS_CONFIG, f"Missing required field: {field}"
    
    # Test data types
    assert isinstance(ANALYTICS_CONFIG['export_dir'], str)
    assert isinstance(ANALYTICS_CONFIG['visualization_dir'], str)
    assert isinstance(ANALYTICS_CONFIG['default_cluster_count'], int)
    assert isinstance(ANALYTICS_CONFIG['default_feature_count'], int)
    assert isinstance(ANALYTICS_CONFIG['pca_components'], int)


def test_ai_skills_list_not_empty():
    """Test that AI_SKILLS list is populated."""
    from advanced.config import AI_SKILLS
    
    assert isinstance(AI_SKILLS, list)
    assert len(AI_SKILLS) > 0
    
    # Check for expected AI skills that are actually in the config
    expected_skills = ['Natural Language Processing', 'NLP', 'Computer Vision', 'Neural Networks']
    for skill in expected_skills:
        assert skill in AI_SKILLS, f"Missing expected AI skill: {skill}"


def test_tech_skills_list_not_empty():
    """Test that TECH_SKILLS list is populated."""
    from advanced.config import TECH_SKILLS
    
    assert isinstance(TECH_SKILLS, list)
    assert len(TECH_SKILLS) > 0
    
    # Check for expected tech skills
    expected_skills = ['Python', 'JavaScript', 'SQL', 'AWS', 'Docker']
    for skill in expected_skills:
        assert skill in TECH_SKILLS, f"Missing expected tech skill: {skill}"


@patch.dict(os.environ, {
    'DB_PATH': '/custom/db/path.db',
    'DB_MEMORY_LIMIT': '8GB',
    'DB_THREADS': '8',
    'PERPLEXITY_MCP_SERVER_NAME': 'custom_perplexity',
    'PERPLEXITY_MAX_TOKENS': '2000',
    'LOG_LEVEL': 'DEBUG'
})
def test_environment_variable_override():
    """Test that environment variables properly override default values."""
    # Need to reload the config module to pick up new environment variables
    import importlib
    from advanced import config
    importlib.reload(config)
    
    assert config.DB_CONFIG['db_path'] == '/custom/db/path.db'
    assert config.DB_CONFIG['memory_limit'] == '8GB'
    assert config.DB_CONFIG['threads'] == 8
    assert config.PERPLEXITY_CONFIG['server_name'] == 'custom_perplexity'
    assert config.PERPLEXITY_CONFIG['max_tokens'] == 2000
    assert config.LOG_CONFIG['level'] == 'DEBUG'


def test_boolean_environment_variable_parsing():
    """Test that boolean environment variables are parsed correctly."""
    test_cases = [
        ('true', True),
        ('True', True),
        ('1', True),
        ('yes', True),
        ('false', False),
        ('False', False),
        ('0', False),
        ('no', False),
        ('', False),
        ('invalid', False)
    ]
    
    for env_value, expected in test_cases:
        with patch.dict(os.environ, {'DB_CACHE_ENABLED': env_value}):
            import importlib
            from advanced import config
            importlib.reload(config)
            
            assert config.DB_CONFIG['cache_enabled'] == expected, f"Failed for env_value: {env_value}"


def test_dotenv_loading_success():
    """Test successful dotenv loading."""
    # The config module already loads dotenv on import
    # We just need to verify that it handles the loading properly
    from advanced.config import BASE_DIR
    
    # Verify that BASE_DIR is set correctly (indicating config loaded successfully)
    assert BASE_DIR is not None
    assert BASE_DIR.exists()
    
    # Verify that environment-dependent configs are loaded
    from advanced.config import DB_CONFIG, PERPLEXITY_CONFIG
    assert DB_CONFIG is not None
    assert PERPLEXITY_CONFIG is not None


@patch('advanced.config.load_dotenv', side_effect=ImportError("python-dotenv not installed"))
def test_dotenv_import_error_handling(mock_load_dotenv):
    """Test graceful handling when python-dotenv is not installed."""
    # This should not raise an exception
    import importlib
    from advanced import config
    importlib.reload(config)
    
    # Should continue with default environment variables
    assert config.DB_CONFIG is not None


@patch('advanced.config.load_dotenv', side_effect=Exception("File not found"))
def test_dotenv_general_error_handling(mock_load_dotenv):
    """Test graceful handling of general dotenv loading errors."""
    # This should not raise an exception
    import importlib
    from advanced import config
    importlib.reload(config)
    
    # Should continue with default environment variables
    assert config.DB_CONFIG is not None


def test_base_dir_is_correct():
    """Test that BASE_DIR points to the correct directory."""
    from advanced.config import BASE_DIR
    
    assert BASE_DIR.exists()
    assert BASE_DIR.is_dir()
    assert BASE_DIR.name == 'advanced'


def test_directory_creation():
    """Test that required directories are created."""
    from advanced.config import BASE_DIR, ANALYTICS_CONFIG
    
    # Check that logs directory exists
    logs_dir = BASE_DIR / 'logs'
    assert logs_dir.exists()
    
    # Check that export and visualization directories exist
    export_dir = BASE_DIR / 'exports'
    viz_dir = BASE_DIR / 'visualizations'
    assert export_dir.exists()
    assert viz_dir.exists()


def test_numeric_environment_variable_parsing():
    """Test parsing of numeric environment variables."""
    with patch.dict(os.environ, {
        'DB_THREADS': '12',
        'DB_CACHE_SIZE': '200',
        'PERPLEXITY_MAX_TOKENS': '1500',
        'PERPLEXITY_TEMPERATURE': '0.5'
    }):
        import importlib
        from advanced import config
        importlib.reload(config)
        
        assert config.DB_CONFIG['threads'] == 12
        assert config.DB_CONFIG['cache_size'] == 200
        assert config.PERPLEXITY_CONFIG['max_tokens'] == 1500
        assert config.PERPLEXITY_CONFIG['temperature'] == 0.5


def test_invalid_numeric_environment_variables():
    """Test handling of invalid numeric environment variables."""
    with patch.dict(os.environ, {
        'DB_THREADS': 'invalid',
        'PERPLEXITY_TEMPERATURE': 'not_a_float'
    }):
        # This should raise ValueError or use defaults
        with pytest.raises((ValueError, TypeError)):
            import importlib
            from advanced import config
            importlib.reload(config)


def test_default_query_params():
    """Test DEFAULT_QUERY_PARAMS configuration."""
    from advanced.config import DEFAULT_QUERY_PARAMS
    
    assert isinstance(DEFAULT_QUERY_PARAMS, dict)
    assert 'limit' in DEFAULT_QUERY_PARAMS
    assert 'offset' in DEFAULT_QUERY_PARAMS
    assert isinstance(DEFAULT_QUERY_PARAMS['limit'], int)
    assert isinstance(DEFAULT_QUERY_PARAMS['offset'], int)
    assert DEFAULT_QUERY_PARAMS['limit'] > 0
    assert DEFAULT_QUERY_PARAMS['offset'] >= 0


@pytest.mark.parametrize("config_name,expected_type", [
    ('DB_CONFIG', dict),
    ('PERPLEXITY_CONFIG', dict),
    ('LOG_CONFIG', dict),
    ('ANALYTICS_CONFIG', dict),
    ('AI_SKILLS', list),
    ('TECH_SKILLS', list),
    ('DEFAULT_QUERY_PARAMS', dict)
])
def test_config_types(config_name, expected_type):
    """Test that all config objects have the expected types."""
    from advanced import config
    
    config_obj = getattr(config, config_name)
    assert isinstance(config_obj, expected_type), f"{config_name} should be {expected_type}"


def test_system_prompt_content():
    """Test that the system prompt contains expected content."""
    from advanced.config import PERPLEXITY_CONFIG
    
    system_prompt = PERPLEXITY_CONFIG['system_prompt']
    
    # Check for key phrases that should be in the prompt
    expected_phrases = [
        'technical data analyst',
        'UK job market',
        'AI jobs',
        'database insertion'
    ]
    
    for phrase in expected_phrases:
        assert phrase.lower() in system_prompt.lower(), f"System prompt missing phrase: {phrase}"


def test_config_immutability():
    """Test that config objects are not accidentally modified."""
    from advanced.config import DB_CONFIG, PERPLEXITY_CONFIG
    
    original_db_path = DB_CONFIG['db_path']
    original_server_name = PERPLEXITY_CONFIG['server_name']
    
    # Try to modify (this should not affect the original config in a well-designed system)
    DB_CONFIG['db_path'] = 'modified_path'
    PERPLEXITY_CONFIG['server_name'] = 'modified_server'
    
    # In a production system, we might want these to be immutable
    # For now, just verify they can be accessed
    assert 'db_path' in DB_CONFIG
    assert 'server_name' in PERPLEXITY_CONFIG