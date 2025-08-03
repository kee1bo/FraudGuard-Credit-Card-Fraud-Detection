"""
Audit Logger for Feature Mapping Operations
Comprehensive logging system for tracking all feature mapping operations,
maintaining audit trails for compliance and monitoring purposes.
"""

import json
import csv
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import asdict
import hashlib
import uuid
import pandas as pd

from fraudguard.entity.feature_mapping_entity import (
    UserTransactionInput, MappingResult, MappingExplanation, QualityMetrics
)
from fraudguard.logger import fraud_logger


class AuditLogger:
    """Comprehensive audit logging for feature mapping operations"""
    
    def __init__(self, 
                 audit_db_path: str = "artifacts/audit/mapping_audit.db",
                 json_log_path: str = "artifacts/audit/mapping_operations.jsonl",
                 csv_log_path: str = "artifacts/audit/mapping_summary.csv"):
        
        self.audit_db_path = Path(audit_db_path)
        self.json_log_path = Path(json_log_path)
        self.csv_log_path = Path(csv_log_path)
        
        # Create directories
        self.audit_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Initialize CSV if it doesn't exist
        self._initialize_csv()
    
    def _initialize_database(self):
        """Initialize SQLite database for audit logging"""
        try:
            with sqlite3.connect(self.audit_db_path) as conn:
                cursor = conn.cursor()
                
                # Create main audit table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS mapping_audit (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        session_id TEXT,
                        user_id TEXT,
                        operation_type TEXT NOT NULL,
                        input_hash TEXT NOT NULL,
                        mapper_type TEXT NOT NULL,
                        model_version TEXT,
                        processing_time_ms REAL,
                        confidence_score REAL,
                        fraud_prediction INTEGER,
                        fraud_probability REAL,
                        validation_status TEXT,
                        error_message TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create input details table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS input_details (
                        audit_id TEXT,
                        transaction_amount REAL,
                        merchant_category TEXT,
                        hour_of_day INTEGER,
                        day_of_week INTEGER,
                        is_weekend INTEGER,
                        location_risk TEXT,
                        spending_pattern TEXT,
                        FOREIGN KEY (audit_id) REFERENCES mapping_audit (id)
                    )
                ''')
                
                # Create mapping results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS mapping_results (
                        audit_id TEXT,
                        pca_component TEXT,
                        estimated_value REAL,
                        confidence_lower REAL,
                        confidence_upper REAL,
                        FOREIGN KEY (audit_id) REFERENCES mapping_audit (id)
                    )
                ''')
                
                # Create quality metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        audit_id TEXT,
                        correlation_preservation REAL,
                        distribution_similarity REAL,
                        prediction_consistency REAL,
                        mapping_uncertainty REAL,
                        overall_confidence REAL,
                        FOREIGN KEY (audit_id) REFERENCES mapping_audit (id)
                    )
                ''')
                
                # Create feature contributions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feature_contributions (
                        audit_id TEXT,
                        feature_name TEXT,
                        contribution_value REAL,
                        FOREIGN KEY (audit_id) REFERENCES mapping_audit (id)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON mapping_audit (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_mapper_type ON mapping_audit (mapper_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON mapping_audit (confidence_score)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_fraud_prediction ON mapping_audit (fraud_prediction)')
                
                conn.commit()
                fraud_logger.info("Audit database initialized successfully")
                
        except Exception as e:
            fraud_logger.error(f"Error initializing audit database: {e}")
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not self.csv_log_path.exists():
            try:
                headers = [
                    'audit_id', 'timestamp', 'operation_type', 'mapper_type',
                    'transaction_amount', 'merchant_category', 'confidence_score',
                    'fraud_prediction', 'fraud_probability', 'processing_time_ms',
                    'validation_status'
                ]
                
                with open(self.csv_log_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)
                
                fraud_logger.info("CSV audit log initialized")
                
            except Exception as e:
                fraud_logger.error(f"Error initializing CSV audit log: {e}")
    
    def log_mapping_operation(self,
                            user_input: UserTransactionInput,
                            mapping_result: Optional[Dict[str, Any]] = None,
                            quality_metrics: Optional[QualityMetrics] = None,
                            explanation: Optional[MappingExplanation] = None,
                            session_id: Optional[str] = None,
                            user_id: Optional[str] = None,
                            error_message: Optional[str] = None) -> str:
        """
        Log a complete mapping operation
        
        Args:
            user_input: Original user input
            mapping_result: Result of the mapping operation
            quality_metrics: Quality assessment metrics
            explanation: Mapping explanation
            session_id: Optional session identifier
            user_id: Optional user identifier
            error_message: Optional error message if operation failed
            
        Returns:
            Audit ID for the logged operation
        """
        audit_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            # Create input hash for deduplication and integrity
            input_hash = self._create_input_hash(user_input)
            
            # Extract key information
            operation_type = "feature_mapping"
            mapper_type = explanation.mapping_method if explanation else "unknown"
            processing_time_ms = mapping_result.get('processing_time_ms', 0) if mapping_result else 0
            confidence_score = mapping_result.get('mapping_confidence', 0) if mapping_result else 0
            fraud_prediction = mapping_result.get('prediction', -1) if mapping_result else -1
            fraud_probability = mapping_result.get('fraud_probability', 0) if mapping_result else 0
            validation_status = "success" if not error_message else "error"
            
            # Log to database
            self._log_to_database(
                audit_id, timestamp, session_id, user_id, operation_type,
                input_hash, mapper_type, processing_time_ms, confidence_score,
                fraud_prediction, fraud_probability, validation_status,
                error_message, user_input, mapping_result, quality_metrics, explanation
            )
            
            # Log to JSON file
            self._log_to_json(
                audit_id, timestamp, user_input, mapping_result,
                quality_metrics, explanation, session_id, user_id, error_message
            )
            
            # Log to CSV
            self._log_to_csv(
                audit_id, timestamp, operation_type, mapper_type,
                user_input, confidence_score, fraud_prediction,
                fraud_probability, processing_time_ms, validation_status
            )
            
            fraud_logger.info(f"Mapping operation logged with audit ID: {audit_id}")
            return audit_id
            
        except Exception as e:
            fraud_logger.error(f"Error logging mapping operation: {e}")
            return audit_id
    
    def _create_input_hash(self, user_input: UserTransactionInput) -> str:
        """Create hash of user input for integrity checking"""
        input_dict = {
            'amount': user_input.transaction_amount,
            'merchant': user_input.merchant_category.value,
            'hour': user_input.time_context.hour_of_day,
            'day': user_input.time_context.day_of_week,
            'weekend': user_input.time_context.is_weekend,
            'location': user_input.location_risk.value,
            'spending': user_input.spending_pattern.value
        }
        
        input_str = json.dumps(input_dict, sort_keys=True)
        return hashlib.sha256(input_str.encode()).hexdigest()
    
    def _log_to_database(self, audit_id: str, timestamp: str, session_id: Optional[str],
                        user_id: Optional[str], operation_type: str, input_hash: str,
                        mapper_type: str, processing_time_ms: float, confidence_score: float,
                        fraud_prediction: int, fraud_probability: float, validation_status: str,
                        error_message: Optional[str], user_input: UserTransactionInput,
                        mapping_result: Optional[Dict], quality_metrics: Optional[QualityMetrics],
                        explanation: Optional[MappingExplanation]):
        """Log to SQLite database"""
        try:
            with sqlite3.connect(self.audit_db_path) as conn:
                cursor = conn.cursor()
                
                # Insert main audit record
                cursor.execute('''
                    INSERT INTO mapping_audit (
                        id, timestamp, session_id, user_id, operation_type,
                        input_hash, mapper_type, processing_time_ms, confidence_score,
                        fraud_prediction, fraud_probability, validation_status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    audit_id, timestamp, session_id, user_id, operation_type,
                    input_hash, mapper_type, processing_time_ms, confidence_score,
                    fraud_prediction, fraud_probability, validation_status, error_message
                ))
                
                # Insert input details
                cursor.execute('''
                    INSERT INTO input_details (
                        audit_id, transaction_amount, merchant_category, hour_of_day,
                        day_of_week, is_weekend, location_risk, spending_pattern
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    audit_id, user_input.transaction_amount, user_input.merchant_category.value,
                    user_input.time_context.hour_of_day, user_input.time_context.day_of_week,
                    int(user_input.time_context.is_weekend), user_input.location_risk.value,
                    user_input.spending_pattern.value
                ))
                
                # Insert mapping results if available
                if explanation and explanation.pca_estimates:
                    for component, value in explanation.pca_estimates.items():
                        confidence_interval = explanation.confidence_intervals.get(component, (0, 0))
                        cursor.execute('''
                            INSERT INTO mapping_results (
                                audit_id, pca_component, estimated_value,
                                confidence_lower, confidence_upper
                            ) VALUES (?, ?, ?, ?, ?)
                        ''', (
                            audit_id, component, value,
                            confidence_interval[0], confidence_interval[1]
                        ))
                
                # Insert quality metrics if available
                if quality_metrics:
                    cursor.execute('''
                        INSERT INTO quality_metrics (
                            audit_id, correlation_preservation, distribution_similarity,
                            prediction_consistency, mapping_uncertainty, overall_confidence
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        audit_id, quality_metrics.correlation_preservation,
                        quality_metrics.distribution_similarity, quality_metrics.prediction_consistency,
                        quality_metrics.mapping_uncertainty, quality_metrics.confidence_score
                    ))
                
                # Insert feature contributions if available
                if explanation and explanation.input_contributions:
                    for feature, contribution in explanation.input_contributions.items():
                        cursor.execute('''
                            INSERT INTO feature_contributions (
                                audit_id, feature_name, contribution_value
                            ) VALUES (?, ?, ?)
                        ''', (audit_id, feature, contribution))
                
                conn.commit()
                
        except Exception as e:
            fraud_logger.error(f"Error logging to database: {e}")
    
    def _log_to_json(self, audit_id: str, timestamp: str, user_input: UserTransactionInput,
                    mapping_result: Optional[Dict], quality_metrics: Optional[QualityMetrics],
                    explanation: Optional[MappingExplanation], session_id: Optional[str],
                    user_id: Optional[str], error_message: Optional[str]):
        """Log to JSON Lines file"""
        try:
            log_entry = {
                'audit_id': audit_id,
                'timestamp': timestamp,
                'session_id': session_id,
                'user_id': user_id,
                'user_input': {
                    'transaction_amount': user_input.transaction_amount,
                    'merchant_category': user_input.merchant_category.value,
                    'time_context': {
                        'hour_of_day': user_input.time_context.hour_of_day,
                        'day_of_week': user_input.time_context.day_of_week,
                        'is_weekend': user_input.time_context.is_weekend,
                        'is_holiday': user_input.time_context.is_holiday
                    },
                    'location_risk': user_input.location_risk.value,
                    'spending_pattern': user_input.spending_pattern.value
                },
                'mapping_result': mapping_result,
                'quality_metrics': asdict(quality_metrics) if quality_metrics else None,
                'explanation': asdict(explanation) if explanation else None,
                'error_message': error_message
            }
            
            with open(self.json_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            fraud_logger.error(f"Error logging to JSON: {e}")
    
    def _log_to_csv(self, audit_id: str, timestamp: str, operation_type: str,
                   mapper_type: str, user_input: UserTransactionInput,
                   confidence_score: float, fraud_prediction: int,
                   fraud_probability: float, processing_time_ms: float,
                   validation_status: str):
        """Log summary to CSV file"""
        try:
            row = [
                audit_id, timestamp, operation_type, mapper_type,
                user_input.transaction_amount, user_input.merchant_category.value,
                confidence_score, fraud_prediction, fraud_probability,
                processing_time_ms, validation_status
            ]
            
            with open(self.csv_log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
                
        except Exception as e:
            fraud_logger.error(f"Error logging to CSV: {e}")
    
    def get_audit_records(self, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         mapper_type: Optional[str] = None,
                         limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve audit records with optional filtering
        
        Args:
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            mapper_type: Filter by mapper type
            limit: Maximum number of records to return
            
        Returns:
            List of audit records
        """
        try:
            with sqlite3.connect(self.audit_db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()
                
                query = '''
                    SELECT ma.*, id.*, qm.correlation_preservation, qm.distribution_similarity,
                           qm.prediction_consistency, qm.mapping_uncertainty, qm.overall_confidence
                    FROM mapping_audit ma
                    LEFT JOIN input_details id ON ma.id = id.audit_id
                    LEFT JOIN quality_metrics qm ON ma.id = qm.audit_id
                    WHERE 1=1
                '''
                params = []
                
                if start_date:
                    query += ' AND ma.timestamp >= ?'
                    params.append(start_date)
                
                if end_date:
                    query += ' AND ma.timestamp <= ?'
                    params.append(end_date)
                
                if mapper_type:
                    query += ' AND ma.mapper_type = ?'
                    params.append(mapper_type)
                
                query += ' ORDER BY ma.timestamp DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            fraud_logger.error(f"Error retrieving audit records: {e}")
            return []
    
    def get_audit_statistics(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get audit statistics for a given time period
        
        Args:
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            
        Returns:
            Dictionary with audit statistics
        """
        try:
            with sqlite3.connect(self.audit_db_path) as conn:
                cursor = conn.cursor()
                
                # Base query conditions
                where_clause = "WHERE 1=1"
                params = []
                
                if start_date:
                    where_clause += " AND timestamp >= ?"
                    params.append(start_date)
                
                if end_date:
                    where_clause += " AND timestamp <= ?"
                    params.append(end_date)
                
                # Total operations
                cursor.execute(f"SELECT COUNT(*) FROM mapping_audit {where_clause}", params)
                total_operations = cursor.fetchone()[0]
                
                # Operations by mapper type
                cursor.execute(f'''
                    SELECT mapper_type, COUNT(*) as count 
                    FROM mapping_audit {where_clause}
                    GROUP BY mapper_type
                ''', params)
                mapper_stats = dict(cursor.fetchall())
                
                # Success/error rates
                cursor.execute(f'''
                    SELECT validation_status, COUNT(*) as count
                    FROM mapping_audit {where_clause}
                    GROUP BY validation_status
                ''', params)
                status_stats = dict(cursor.fetchall())
                
                # Average confidence score
                cursor.execute(f'''
                    SELECT AVG(confidence_score) 
                    FROM mapping_audit {where_clause}
                    AND confidence_score IS NOT NULL
                ''', params)
                avg_confidence = cursor.fetchone()[0] or 0
                
                # Average processing time
                cursor.execute(f'''
                    SELECT AVG(processing_time_ms)
                    FROM mapping_audit {where_clause}
                    AND processing_time_ms IS NOT NULL
                ''', params)
                avg_processing_time = cursor.fetchone()[0] or 0
                
                # Fraud detection stats
                cursor.execute(f'''
                    SELECT fraud_prediction, COUNT(*) as count
                    FROM mapping_audit {where_clause}
                    AND fraud_prediction != -1
                    GROUP BY fraud_prediction
                ''', params)
                fraud_stats = dict(cursor.fetchall())
                
                return {
                    'total_operations': total_operations,
                    'mapper_statistics': mapper_stats,
                    'status_statistics': status_stats,
                    'average_confidence_score': round(avg_confidence, 3),
                    'average_processing_time_ms': round(avg_processing_time, 2),
                    'fraud_detection_statistics': fraud_stats,
                    'success_rate': round(status_stats.get('success', 0) / max(total_operations, 1) * 100, 2)
                }
                
        except Exception as e:
            fraud_logger.error(f"Error getting audit statistics: {e}")
            return {}
    
    def export_audit_data(self, 
                         output_path: str,
                         format: str = 'csv',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> bool:
        """
        Export audit data to file
        
        Args:
            output_path: Path for output file
            format: Export format ('csv', 'json', 'excel')
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            records = self.get_audit_records(start_date, end_date, limit=10000)
            
            if not records:
                fraud_logger.warning("No audit records found for export")
                return False
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'csv':
                df = pd.DataFrame(records)
                df.to_csv(output_file, index=False)
            
            elif format.lower() == 'json':
                with open(output_file, 'w') as f:
                    json.dump(records, f, indent=2, default=str)
            
            elif format.lower() == 'excel':
                df = pd.DataFrame(records)
                df.to_excel(output_file, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            fraud_logger.info(f"Audit data exported to {output_file}")
            return True
            
        except Exception as e:
            fraud_logger.error(f"Error exporting audit data: {e}")
            return False
    
    def cleanup_old_records(self, days_to_keep: int = 90) -> int:
        """
        Clean up old audit records
        
        Args:
            days_to_keep: Number of days of records to keep
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - pd.Timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            with sqlite3.connect(self.audit_db_path) as conn:
                cursor = conn.cursor()
                
                # Get count of records to be deleted
                cursor.execute(
                    'SELECT COUNT(*) FROM mapping_audit WHERE timestamp < ?',
                    (cutoff_str,)
                )
                count_to_delete = cursor.fetchone()[0]
                
                if count_to_delete > 0:
                    # Delete from all related tables
                    cursor.execute(
                        'DELETE FROM feature_contributions WHERE audit_id IN (SELECT id FROM mapping_audit WHERE timestamp < ?)',
                        (cutoff_str,)
                    )
                    cursor.execute(
                        'DELETE FROM quality_metrics WHERE audit_id IN (SELECT id FROM mapping_audit WHERE timestamp < ?)',
                        (cutoff_str,)
                    )
                    cursor.execute(
                        'DELETE FROM mapping_results WHERE audit_id IN (SELECT id FROM mapping_audit WHERE timestamp < ?)',
                        (cutoff_str,)
                    )
                    cursor.execute(
                        'DELETE FROM input_details WHERE audit_id IN (SELECT id FROM mapping_audit WHERE timestamp < ?)',
                        (cutoff_str,)
                    )
                    cursor.execute(
                        'DELETE FROM mapping_audit WHERE timestamp < ?',
                        (cutoff_str,)
                    )
                    
                    conn.commit()
                    fraud_logger.info(f"Cleaned up {count_to_delete} old audit records")
                
                return count_to_delete
                
        except Exception as e:
            fraud_logger.error(f"Error cleaning up old records: {e}")
            return 0