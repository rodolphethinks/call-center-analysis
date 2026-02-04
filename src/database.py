"""
Database module for storing and retrieving call center analytics data
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import sqlite3
import json
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class CallCenterDatabase:
    """
    SQLite database for call center analytics
    """
    
    def __init__(self, db_path: str = "data/database/analytics.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
        logger.info(f"Database initialized: {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Calls table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT UNIQUE NOT NULL,
                    call_datetime DATETIME,
                    duration REAL,
                    transcription TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT UNIQUE NOT NULL,
                    customer_summary TEXT,
                    agent_summary TEXT,
                    customer_sentiment_trajectory TEXT,
                    customer_sentiment_examples TEXT,
                    key_issues TEXT,
                    agent_performance_overall TEXT,
                    agent_understanding TEXT,
                    resolution_status TEXT,
                    improvement_suggestions TEXT,
                    analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (call_id) REFERENCES calls(call_id)
                )
            """)
            
            # Issues table (normalized)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL,
                    category TEXT,
                    issue_description TEXT,
                    FOREIGN KEY (call_id) REFERENCES calls(call_id)
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_call_datetime ON calls(call_datetime)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON analysis(customer_sentiment_trajectory)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_resolution ON analysis(resolution_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_issue_category ON issues(category)")
    
    def insert_call(
        self,
        call_id: str,
        transcription: str,
        call_datetime: Optional[datetime] = None,
        duration: Optional[float] = None
    ) -> bool:
        """Insert a call record"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO calls (call_id, call_datetime, duration, transcription)
                    VALUES (?, ?, ?, ?)
                """, (call_id, call_datetime, duration, transcription))
            return True
        except Exception as e:
            logger.error(f"Failed to insert call {call_id}: {e}")
            return False
    
    def insert_analysis(self, call_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Insert analysis results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Extract analysis from JSON string if needed
                if isinstance(analysis_data, str):
                    try:
                        analysis_data = json.loads(analysis_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse analysis data for {call_id}")
                        return False
                
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis (
                        call_id, customer_summary, agent_summary,
                        customer_sentiment_trajectory, customer_sentiment_examples,
                        key_issues, agent_performance_overall, agent_understanding,
                        resolution_status, improvement_suggestions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    call_id,
                    analysis_data.get('customer_summary'),
                    analysis_data.get('agent_summary'),
                    analysis_data.get('customer_sentiment_trajectory'),
                    analysis_data.get('customer_sentiment_examples'),
                    json.dumps(analysis_data.get('key_issues', [])),
                    analysis_data.get('agent_performance_overall'),
                    analysis_data.get('agent_understanding'),
                    analysis_data.get('resolution_status'),
                    analysis_data.get('improvement_suggestions')
                ))
            
            return True
        except Exception as e:
            logger.error(f"Failed to insert analysis for {call_id}: {e}")
            return False
    
    def insert_issues(self, call_id: str, issues: Dict[str, List[str]]) -> bool:
        """Insert categorized issues"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete existing issues for this call
                cursor.execute("DELETE FROM issues WHERE call_id = ?", (call_id,))
                
                # Insert new issues
                for category, issue_list in issues.items():
                    for issue in issue_list:
                        cursor.execute("""
                            INSERT INTO issues (call_id, category, issue_description)
                            VALUES (?, ?, ?)
                        """, (call_id, category, issue))
            
            return True
        except Exception as e:
            logger.error(f"Failed to insert issues for {call_id}: {e}")
            return False
    
    def get_call(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve call data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_analysis(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analysis WHERE call_id = ?", (call_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                # Parse JSON fields
                if data.get('key_issues'):
                    try:
                        data['key_issues'] = json.loads(data['key_issues'])
                    except:
                        pass
                return data
            return None
    
    def get_all_calls(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve all calls"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM calls ORDER BY call_datetime DESC"
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total calls
            cursor.execute("SELECT COUNT(*) as count FROM calls")
            stats['total_calls'] = cursor.fetchone()['count']
            
            # Calls with analysis
            cursor.execute("SELECT COUNT(*) as count FROM analysis")
            stats['analyzed_calls'] = cursor.fetchone()['count']
            
            # Sentiment distribution
            cursor.execute("""
                SELECT customer_sentiment_trajectory, COUNT(*) as count
                FROM analysis
                GROUP BY customer_sentiment_trajectory
            """)
            stats['sentiment_distribution'] = {
                row['customer_sentiment_trajectory']: row['count']
                for row in cursor.fetchall()
            }
            
            # Resolution distribution
            cursor.execute("""
                SELECT resolution_status, COUNT(*) as count
                FROM analysis
                GROUP BY resolution_status
            """)
            stats['resolution_distribution'] = {
                row['resolution_status']: row['count']
                for row in cursor.fetchall()
            }
            
            # Issue categories
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM issues
                GROUP BY category
            """)
            stats['issue_categories'] = {
                row['category']: row['count']
                for row in cursor.fetchall()
            }
            
            return stats
    
    def export_to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export entire database to dictionary"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            data = {}
            
            # Export calls
            cursor.execute("SELECT * FROM calls")
            data['calls'] = [dict(row) for row in cursor.fetchall()]
            
            # Export analysis
            cursor.execute("SELECT * FROM analysis")
            data['analysis'] = [dict(row) for row in cursor.fetchall()]
            
            # Export issues
            cursor.execute("SELECT * FROM issues")
            data['issues'] = [dict(row) for row in cursor.fetchall()]
            
            return data


def create_database(config: Dict) -> CallCenterDatabase:
    """
    Factory function to create CallCenterDatabase from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CallCenterDatabase instance
    """
    db_config = config.get('database', {})
    db_path = db_config.get('path', 'data/database/analytics.db')
    
    return CallCenterDatabase(db_path)
