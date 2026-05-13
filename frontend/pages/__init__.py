"""
Pages module - contains all page widgets
"""
from .dashboard_page import DashboardPage
from .camera_page import CameraMonitorPage
from .search_page import PersonSearchPage
from .registration_page import RegistrationPage
from .reports_page import ReportsPage
from .settings_page import SettingsPage

__all__ = [
    'DashboardPage',
    'CameraMonitorPage',
    'PersonSearchPage',
    'RegistrationPage',
    'ReportsPage',
    'SettingsPage',
]
