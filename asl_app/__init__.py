"""Modulo ASL Recognition App."""
import reflex as rx
from asl_app.state import ASLState
from asl_app.pages.index import index

# Crear la aplicacion
app = rx.App()

# Agregar pagina principal
app.add_page(index, route="/", title="ASL Recognizer")
