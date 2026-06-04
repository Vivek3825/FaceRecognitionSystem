"""
Pages module - contains all page widgets
"""
from .dashboard_page import DashboardPage
from .camera_page import CameraMonitorPage
from .registration_page import RegistrationPage
# from ..rough.search_page import PersonSearchPage
# from ..rough.reports_page import ReportsPage
from .settings_page import SettingsPage

__all__ = [
    'DashboardPage',
    'CameraMonitorPage',
    'RegistrationPage',
    #'PersonSearchPage',
    #'ReportsPage',
    'SettingsPage',
]
