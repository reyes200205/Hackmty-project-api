# Project API (FastAPI)

API construida con [FastAPI](https://fastapi.tiangolo.com/).

---

## 📋 Requisitos Previos

- **Python 3.10+** instalado en tu sistema.
- Gestor de paquetes **pip**.

---

## 🚀 Guía de Instalación y Ejecución

Sigue estos pasos para clonar, configurar y ejecutar el proyecto localmente:

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU-USUARIO/project-api.git
cd project-api
```

### 2. Crear un entorno virtual

Crea un entorno virtual para aislar las dependencias del proyecto:

- **En Windows:**
  ```powershell
  python -m venv .venv
  ```

- **En macOS / Linux:**
  ```bash
  python3 -m venv .venv
  ```

---

### 3. Activar el entorno virtual

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(Si recibes un error sobre directivas de ejecución de scripts, ejecuta antes: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*

- **Windows (Símbolo del sistema / CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **macOS / Linux (Bash / Zsh):**
  ```bash
  source .venv/bin/activate
  ```

> Una vez activado, verás `(.venv)` al inicio de la línea de comandos en tu terminal.

---

### 4. Instalar dependencias

Con el entorno virtual activado, instala los paquetes requeridos:

```bash
pip install -r requirements.txt
```

---

### 5. Ejecutar la aplicación

Inicia el servidor en modo desarrollo con recarga automática:

```bash
fastapi dev main.py
```

O alternativamente usando Uvicorn:

```bash
uvicorn main:app --reload --reload-exclude ".venv/*"
```

> ⚠️ Sin `--reload-exclude ".venv/*"`, instalar cualquier paquete nuevo (pip toca miles de archivos dentro de `.venv`) dispara un reinicio completo del servidor y corta cualquier llamada de Twilio en curso. Para la demo en vivo con jueces, mejor correr **sin** `--reload` (`uvicorn main:app`), para cero riesgo de reinicio accidental.

---

## 🌐 Endpoints y Documentación

Una vez levantado el servidor:

- **API Base:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Documentación Interactiva (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Documentación Alternativa (ReDoc):** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📁 Estructura del Proyecto

```text
project-api/
│
├── .gitignore          # Archivos y carpetas ignorados por git (ej. .venv)
├── main.py             # Punto de entrada de la aplicación FastAPI
├── README.md           # Documentación e instrucciones del proyecto
└── requirements.txt    # Dependencias del proyecto
```
