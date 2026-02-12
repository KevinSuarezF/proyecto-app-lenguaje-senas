"""Configuración de Reflex para la aplicación de reconocimiento de señas ASL."""

import reflex as rx

config = rx.Config(
    app_name="asl_app",
    env=rx.Env.DEV,
    frontend_port=3000,
    backend_port=8000,
    tailwind={},
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
)
