"""
Tests for the advanced.data_gatherer module.

These tests verify CLI argument parsing, database initialization,
data gathering workflow, and error handling following TDD principles.
"""

import pytest
import argparse
import sys
from unittest.mock import Mock, patch, MagicMock, call

from advanced.data_gatherer import DataGatherer, main
from advanced.models.data_classes import JobPosting, PerplexityResponse


class TestDataGathererInitialization:
    """Test DataGatherer initialization."""
    
    def test_data_gatherer_initialization_normal_mode(self):
        """Test DataGatherer initializes correctly in normal mode."""
        gatherer = DataGatherer(dry_run=False)
        
        assert gatherer.dry_run == False
        assert hasattr(gatherer, 'client')
        assert hasattr(gatherer, 'processor')
        assert hasattr(gatherer, 'db')
    
    def test_data_gatherer_initialization_dry_run_mode(self):
        """Test DataGatherer initializes correctly in dry-run mode."""
        gatherer = DataGatherer(dry_run=True)
        
        assert gatherer.dry_run == True
        assert hasattr(gatherer, 'client')
        assert hasattr(gatherer, 'processor')
        assert hasattr(gatherer, 'db')
    
    def test_data_gatherer_default_initialization(self):
        """Test DataGatherer default initialization."""
        gatherer = DataGatherer()
        
        assert gatherer.dry_run == False


class TestDatabaseInitialization:
    """Test database initialization methods."""
    
    @patch('advanced.data_gatherer.ALL_TABLES', ['CREATE TABLE test_table (id INTEGER)'])
    @patch('advanced.data_gatherer.INDEX_DEFINITIONS', ['CREATE INDEX test_idx ON test_table(id)'])
    def test_initialize_database_normal_mode(self, mock_database_connection):
        """Test database initialization in normal mode."""
        gatherer = DataGatherer(dry_run=False)
        gatherer.db = mock_database_connection
        
        gatherer.initialize_database()
        
        # Verify database operations were called
        mock_database_connection.__enter__.assert_called()
        mock_database_connection.execute.assert_called()
    
    @patch('advanced.data_gatherer.ALL_TABLES', ['CREATE TABLE IF NOT EXISTS test_table (id INTEGER)'])
    @patch('advanced.data_gatherer.INDEX_DEFINITIONS', ['CREATE INDEX test_idx ON test_table(id)'])
    @patch('advanced.data_gatherer.logger')
    def test_initialize_database_dry_run_mode(self, mock_logger, mock_database_connection):
        """Test database initialization in dry-run mode."""
        gatherer = DataGatherer(dry_run=True)
        gatherer.db = mock_database_connection
        
        gatherer.initialize_database()
        
        # Verify no actual database operations occurred
        mock_database_connection.__enter__.assert_not_called()
        mock_database_connection.execute.assert_not_called()
        
        # Verify logging occurred
        mock_logger.info.assert_called()


class TestDataGathering:
    """Test data gathering methods."""
    
    def test_gather_data_with_custom_query(self):
        """Test data gathering with custom query."""
        gatherer = DataGatherer(dry_run=False)
        custom_query = "What are the latest AI jobs in London?"
        
        result = gatherer.gather_data(query=custom_query)
        
        assert isinstance(result, list)
    
    def test_gather_data_with_parameters(self):
        """Test data gathering with role, location, and timeframe."""
        gatherer = DataGatherer(dry_run=False)
        
        result = gatherer.gather_data(
            role="Data Scientist",
            location="Manchester",
            timeframe="last month",
            batch_size=15
        )
        
        assert isinstance(result, list)
    
    def test_gather_data_dry_run_mode(self):
        """Test data gathering in dry-run mode."""
        gatherer = DataGatherer(dry_run=True)
        
        result = gatherer.gather_data(
            role="ML Engineer",
            location="London"
        )
        
        assert isinstance(result, list)
        assert len(result) == 0  # Dry run returns empty list
    
    def test_gather_data_default_parameters(self):
        """Test data gathering with default parameters."""
        gatherer = DataGatherer(dry_run=False)
        
        result = gatherer.gather_data()
        
        assert isinstance(result, list)


class TestResponseProcessing:
    """Test Perplexity response processing."""
    
    @patch('advanced.data_gatherer.logger')
    def test_process_perplexity_response_valid_data(self, mock_logger, sample_mcp_response):
        """Test processing valid Perplexity response."""
        gatherer = DataGatherer(dry_run=False)
        
        # Mock the client and processor methods
        mock_perplexity_response = Mock()
        mock_job_postings = [Mock(spec=JobPosting)]
        
        gatherer.client.process_response = Mock(return_value=mock_perplexity_response)
        gatherer.client.parse_jobs_data = Mock(return_value=[{'title': 'Test Job'}])
        gatherer.client.transform_to_job_postings = Mock(return_value=mock_job_postings)
        gatherer.processor.process_job_posting = Mock(side_effect=lambda x: x)
        
        result = gatherer.process_perplexity_response(sample_mcp_response)
        
        assert isinstance(result, list)
        assert len(result) == 1
        
        # Verify all processing steps were called
        gatherer.client.process_response.assert_called_once_with(sample_mcp_response)
        gatherer.client.parse_jobs_data.assert_called_once()
        gatherer.client.transform_to_job_postings.assert_called_once()
    
    def test_process_perplexity_response_empty_data(self):
        """Test processing empty Perplexity response."""
        gatherer = DataGatherer(dry_run=False)
        
        # Mock empty response
        empty_response = {"data": {"response": {"choices": []}}}
        
        # Mock the client methods to return empty data
        gatherer.client.process_response = Mock(return_value=Mock())
        gatherer.client.parse_jobs_data = Mock(return_value=[])
        gatherer.client.transform_to_job_postings = Mock(return_value=[])
        
        result = gatherer.process_perplexity_response(empty_response)
        
        assert isinstance(result, list)
        assert len(result) == 0


class TestDatabaseInsertion:
    """Test database insertion methods."""
    
    def test_insert_jobs_to_database_normal_mode(self, sample_job_posting, mock_database_connection):
        """Test inserting jobs to database in normal mode."""
        gatherer = DataGatherer(dry_run=False)
        gatherer.db = mock_database_connection
        job_postings = [sample_job_posting]
        
        result = gatherer.insert_jobs_to_database(job_postings)
        
        assert isinstance(result, int)
        assert result >= 0
        
        # Verify database context manager was used
        mock_database_connection.__enter__.assert_called()
    
    @patch('advanced.data_gatherer.logger')
    def test_insert_jobs_to_database_dry_run_mode(self, mock_logger, sample_job_posting):
        """Test inserting jobs to database in dry-run mode."""
        gatherer = DataGatherer(dry_run=True)
        job_postings = [sample_job_posting]
        
        result = gatherer.insert_jobs_to_database(job_postings)
        
        assert isinstance(result, int)
        assert result == len(job_postings)  # Returns count in dry-run
        
        # Verify logging occurred
        mock_logger.info.assert_called()
    
    def test_insert_jobs_to_database_empty_list(self):
        """Test inserting empty job list."""
        gatherer = DataGatherer(dry_run=False)
        
        result = gatherer.insert_jobs_to_database([])
        
        assert result == 0
    
    def test_insert_job_posting_method(self, sample_job_posting, mock_database_connection):
        """Test the private _insert_job_posting method."""
        gatherer = DataGatherer(dry_run=False)
        gatherer.db = mock_database_connection
        
        # Mock the ai_impact_category as an enum-like object to match expected behavior
        from advanced.models.data_classes import AIImpactLevel
        sample_job_posting.ai_impact_category = AIImpactLevel.HIGH
        
        # This should not raise an exception
        gatherer._insert_job_posting(sample_job_posting)
        
        # Verify execute was called
        mock_database_connection.execute.assert_called()
    
    def test_insert_job_skills_method(self, sample_job_posting, mock_database_connection):
        """Test the private _insert_job_skills method."""
        gatherer = DataGatherer(dry_run=False)
        gatherer.db = mock_database_connection
        
        # This should not raise an exception
        gatherer._insert_job_skills(sample_job_posting)
        
        # Verify execute was called (once per skill)
        assert mock_database_connection.execute.call_count >= len(sample_job_posting.skills)
    
    def test_insert_ai_metrics_method(self, sample_job_posting, mock_database_connection):
        """Test the private _insert_ai_metrics method."""
        gatherer = DataGatherer(dry_run=False)
        gatherer.db = mock_database_connection
        
        # Add mock AI metrics to the job posting
        sample_job_posting.ai_metrics = Mock()
        sample_job_posting.ai_metrics.automation_risk = 0.3
        sample_job_posting.ai_metrics.augmentation_potential = 0.7
        sample_job_posting.ai_metrics.transformation_level = 0.5
        
        # This should not raise an exception
        gatherer._insert_ai_metrics(sample_job_posting)
        
        # Verify execute was called
        mock_database_connection.execute.assert_called()


class TestStatisticsDisplay:
    """Test statistics display methods."""
    
    def test_show_statistics_with_data(self, mock_database_connection):
        """Test showing statistics when data exists."""
        gatherer = DataGatherer(dry_run=False)
        gatherer.db = mock_database_connection
        
        # Mock database responses
        mock_database_connection.fetch_all.side_effect = [
            [(10,)],  # Total jobs count
            [('high', 6), ('medium', 4)],  # AI impact categories
            [('TechCorp', 3), ('AIStart', 2)],  # Top companies
            [('Python', 8), ('TensorFlow', 5)]  # Top skills
        ]
        
        # This should not raise an exception
        gatherer.show_statistics()
        
        # Verify database queries were made
        assert mock_database_connection.fetch_all.call_count >= 4
        mock_database_connection.__enter__.assert_called()
    
    def test_show_statistics_empty_database(self, mock_database_connection):
        """Test showing statistics when no data exists."""
        gatherer = DataGatherer(dry_run=False)
        gatherer.db = mock_database_connection
        
        # Mock empty database responses
        mock_database_connection.fetch_all.side_effect = [
            [(0,)],  # No jobs
            [],      # No categories
            [],      # No companies
            []       # No skills
        ]
        
        # This should not raise an exception
        gatherer.show_statistics()
        
        # Verify database queries were made
        mock_database_connection.fetch_all.assert_called()


class TestErrorHandling:
    """Test error handling in data gatherer."""
    
    def test_insert_jobs_with_exception(self, mock_database_connection):
        """Test that job insertion handles exceptions gracefully."""
        gatherer = DataGatherer(dry_run=False)
        gatherer.db = mock_database_connection
        
        # Create a job posting that will cause an exception
        invalid_job = Mock(spec=JobPosting)
        invalid_job.job_id = None  # This might cause issues
        
        # Mock execute to raise an exception
        mock_database_connection.execute.side_effect = Exception("Database error")
        
        result = gatherer.insert_jobs_to_database([invalid_job])
        
        # Should handle exception and return 0 successful insertions
        assert result == 0
    
    @patch('advanced.data_gatherer.logger')
    def test_error_logging(self, mock_logger, mock_database_connection):
        """Test that errors are properly logged."""
        gatherer = DataGatherer(dry_run=False)
        gatherer.db = mock_database_connection
        
        # Mock execute to raise an exception
        mock_database_connection.execute.side_effect = Exception("Test error")
        
        invalid_job = Mock(spec=JobPosting)
        invalid_job.job_id = "test-error"
        
        gatherer.insert_jobs_to_database([invalid_job])
        
        # Verify error was logged
        mock_logger.error.assert_called()


class TestCLIInterface:
    """Test command-line interface."""
    
    def test_main_function_with_init_flag(self):
        """Test main function with --init flag."""
        test_args = ['--init']
        
        with patch('sys.argv', ['data_gatherer.py'] + test_args):
            with patch('advanced.data_gatherer.DataGatherer') as mock_gatherer_class:
                mock_gatherer = Mock()
                mock_gatherer_class.return_value = mock_gatherer
                
                try:
                    main()
                except SystemExit:
                    pass  # Expected for successful completion
                
                # Verify DataGatherer was created and init was called
                mock_gatherer_class.assert_called_once()
                mock_gatherer.initialize_database.assert_called_once()
    
    def test_main_function_with_gather_flag(self):
        """Test main function with --gather flag."""
        test_args = ['--gather', '--role', 'Data Scientist', '--location', 'London']
        
        with patch('sys.argv', ['data_gatherer.py'] + test_args):
            with patch('advanced.data_gatherer.DataGatherer') as mock_gatherer_class:
                mock_gatherer = Mock()
                mock_gatherer_class.return_value = mock_gatherer
                mock_gatherer.gather_data.return_value = []
                
                try:
                    main()
                except SystemExit:
                    pass  # Expected for successful completion
                
                # Verify gather_data was called with parameters
                mock_gatherer.gather_data.assert_called_once()
                call_args = mock_gatherer.gather_data.call_args
                assert call_args[1]['role'] == 'Data Scientist'
                assert call_args[1]['location'] == 'London'
    
    def test_main_function_with_stats_flag(self):
        """Test main function with --stats flag."""
        test_args = ['--stats']
        
        with patch('sys.argv', ['data_gatherer.py'] + test_args):
            with patch('advanced.data_gatherer.DataGatherer') as mock_gatherer_class:
                mock_gatherer = Mock()
                mock_gatherer_class.return_value = mock_gatherer
                
                try:
                    main()
                except SystemExit:
                    pass  # Expected for successful completion
                
                # Verify show_statistics was called
                mock_gatherer.show_statistics.assert_called_once()
    
    def test_main_function_with_dry_run(self):
        """Test main function with --dry-run flag."""
        test_args = ['--gather', '--dry-run']
        
        with patch('sys.argv', ['data_gatherer.py'] + test_args):
            with patch('advanced.data_gatherer.DataGatherer') as mock_gatherer_class:
                mock_gatherer = Mock()
                mock_gatherer_class.return_value = mock_gatherer
                mock_gatherer.gather_data.return_value = []
                
                try:
                    main()
                except SystemExit:
                    pass  # Expected for successful completion
                
                # Verify DataGatherer was created with dry_run=True
                mock_gatherer_class.assert_called_once_with(dry_run=True)
    
    def test_main_function_no_arguments(self):
        """Test main function with no arguments (should show error)."""
        test_args = []
        
        with patch('sys.argv', ['data_gatherer.py'] + test_args):
            with pytest.raises(SystemExit):
                main()
    
    def test_main_function_keyboard_interrupt(self):
        """Test main function handles KeyboardInterrupt."""
        test_args = ['--init']
        
        with patch('sys.argv', ['data_gatherer.py'] + test_args):
            with patch('advanced.data_gatherer.DataGatherer') as mock_gatherer_class:
                mock_gatherer = Mock()
                mock_gatherer.initialize_database.side_effect = KeyboardInterrupt()
                mock_gatherer_class.return_value = mock_gatherer
                
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                assert exc_info.value.code == 1
    
    def test_main_function_general_exception(self):
        """Test main function handles general exceptions."""
        test_args = ['--init']
        
        with patch('sys.argv', ['data_gatherer.py'] + test_args):
            with patch('advanced.data_gatherer.DataGatherer') as mock_gatherer_class:
                mock_gatherer = Mock()
                mock_gatherer.initialize_database.side_effect = Exception("Test error")
                mock_gatherer_class.return_value = mock_gatherer
                
                with pytest.raises(SystemExit) as exc_info:
                    main()
                
                assert exc_info.value.code == 1
    
    def test_argument_parsing_custom_query(self):
        """Test argument parsing with custom query."""
        test_args = ['--gather', '--query', 'Custom AI jobs query']
        
        with patch('sys.argv', ['data_gatherer.py'] + test_args):
            with patch('advanced.data_gatherer.DataGatherer') as mock_gatherer_class:
                mock_gatherer = Mock()
                mock_gatherer_class.return_value = mock_gatherer
                mock_gatherer.gather_data.return_value = []
                
                try:
                    main()
                except SystemExit:
                    pass
                
                # Verify custom query was passed
                call_args = mock_gatherer.gather_data.call_args
                assert call_args[1]['query'] == 'Custom AI jobs query'
    
    def test_argument_parsing_batch_size(self):
        """Test argument parsing with custom batch size."""
        test_args = ['--gather', '--batch-size', '25']
        
        with patch('sys.argv', ['data_gatherer.py'] + test_args):
            with patch('advanced.data_gatherer.DataGatherer') as mock_gatherer_class:
                mock_gatherer = Mock()
                mock_gatherer_class.return_value = mock_gatherer
                mock_gatherer.gather_data.return_value = []
                
                try:
                    main()
                except SystemExit:
                    pass
                
                # Verify batch size was passed
                call_args = mock_gatherer.gather_data.call_args
                assert call_args[1]['batch_size'] == 25


class TestIntegration:
    """Test integration scenarios."""
    
    @patch('advanced.data_gatherer.logger')
    def test_full_workflow_dry_run(self, mock_logger):
        """Test complete workflow in dry-run mode."""
        gatherer = DataGatherer(dry_run=True)
        # Mock the database for statistics with context manager support
        mock_db = Mock()
        mock_db.fetch_all.side_effect = [
            [(0,)],  # Total jobs count
            [],      # AI impact categories (empty)
            [],      # Top companies (empty)
            []       # Top skills (empty)
        ]
        mock_db.__enter__ = Mock(return_value=mock_db)
        mock_db.__exit__ = Mock(return_value=None)
        gatherer.db = mock_db
        gatherer.db = mock_db
        
        # Test initialization
        gatherer.initialize_database()
        
        # Test data gathering
        jobs = gatherer.gather_data(role="Data Scientist", location="London")
        
        # Test database insertion
        result = gatherer.insert_jobs_to_database(jobs)
        
        # Test statistics
        gatherer.show_statistics()
        
        # All operations should complete without errors
        assert result == 0  # Dry run returns 0
        mock_logger.info.assert_called()  # Should have logged dry-run messages
    
    def test_database_context_manager_usage(self, mock_database_connection):
        """Test that database context manager is used properly."""
        gatherer = DataGatherer(dry_run=False)
        gatherer.db = mock_database_connection
        
        # Test various database operations
        gatherer.initialize_database()
        gatherer.show_statistics()
        
        # Verify context manager was used
        assert mock_database_connection.__enter__.call_count >= 2
        assert mock_database_connection.__exit__.call_count >= 2


class TestLogging:
    """Test logging functionality."""
    
    @patch('advanced.data_gatherer.logger')
    def test_logging_throughout_workflow(self, mock_logger, sample_job_posting):
        """Test that appropriate logging occurs throughout the workflow."""
        gatherer = DataGatherer(dry_run=False)
        
        # Mock database with proper context manager support
        mock_db = Mock()
        mock_db.__enter__ = Mock(return_value=mock_db)
        mock_db.__exit__ = Mock(return_value=None)
        gatherer.db = mock_db
        
        # Fix the ai_impact_category to be a string
        sample_job_posting.ai_impact_category = "high"
        
        # Test various operations
        gatherer.gather_data()
        gatherer.insert_jobs_to_database([sample_job_posting])
        
        # Verify logging occurred
        mock_logger.info.assert_called()
    
    @patch('advanced.data_gatherer.LoggedOperation')
    def test_logged_operation_usage(self, mock_logged_operation):
        """Test that LoggedOperation context manager is used."""
        gatherer = DataGatherer(dry_run=False)
        
        # Test operations that should use LoggedOperation
        gatherer.gather_data()
        
        # Verify LoggedOperation was used
        mock_logged_operation.assert_called()