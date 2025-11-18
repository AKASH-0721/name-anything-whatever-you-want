import os

class Config:
    """Base configuration"""
    PORT = int(os.environ.get('PORT', 8080))
    DB_FOLDER = os.environ.get('DB_FOLDER', 'train')
    EXTRACTED = os.environ.get('EXTRACTED_FOLDER', 'extracted_faces')
    EMB_FILE = os.environ.get('EMB_FILE', 'data/embeddings_db.pkl')
    ATTENDANCE_CSV = os.environ.get('ATTENDANCE_CSV', 'data/attendance.csv')
    SECTION = os.environ.get('SECTION', 'ALL')
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'

class DockerConfig(ProductionConfig):
    """Docker-specific production configuration"""
    EMB_FILE = '/app/data/embeddings_db.pkl'
    ATTENDANCE_CSV = '/app/data/attendance.csv'
    EXTRACTED = '/app/extracted_faces'

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'production')
    if env == 'development':
        return DevelopmentConfig()
    elif os.environ.get('DOCKER_ENV'):
        return DockerConfig()
    else:
        return ProductionConfig()