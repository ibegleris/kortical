import pytest
import sys
sys.path = ["/Users/yiannis/kortical/alex/src"] + sys.path
from alex.main import create_app


@pytest.fixture(scope='session')
def app():
    app = create_app()
    return app
