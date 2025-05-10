import cv2
import os
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox

class DeteccionPersonasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Personas con Clarifai")
        self.ruta_imagen = "imagen_capturada.jpg"
        
        # Credenciales de Clarifai
        self.CLARIFAI_API_KEY = ""  # Tu API Key
        self.CLARIFAI_APP_ID = ""   # Tu APP_ID

        # Inicializar cámara
        self.camara = cv2.VideoCapture(0)
        if not self.camara.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara.")
            self.root.quit()
            return
        self.camara_activa = True

        # Elementos de la interfaz
        self.label_vista_previa = tk.Label(root, text="Vista Previa de la Cámara")
        self.label_vista_previa.pack(pady=5)

        self.label_imagen = tk.Label(root, text="Imagen Capturada")
        self.label_imagen.pack(pady=5)

        self.boton_capturar = tk.Button(root, text="Capturar Imagen", command=self.capturar_imagen)
        self.boton_capturar.pack(pady=5)

        self.boton_procesar = tk.Button(root, text="Procesar Imagen", command=self.procesar_imagen, state=tk.DISABLED)
        self.boton_procesar.pack(pady=5)

        self.texto_resultados = tk.Text(root, height=10, width=50)
        self.texto_resultados.pack(pady=10)
        self.texto_resultados.config(state='disabled')

        # Iniciar vista previa
        self.actualizar_vista_previa()

    def actualizar_vista_previa(self):
        if not self.camara_activa:
            return
        ret, frame = self.camara.read()
        if ret:
            # Convertir frame de OpenCV (BGR) a RGB y luego a imagen Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((400, 300), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.label_vista_previa.config(image=img_tk)
            self.label_vista_previa.image = img_tk
        # Actualizar cada 10 ms
        self.root.after(10, self.actualizar_vista_previa)

    def capturar_imagen(self):
        ret, imagen = self.camara.read()
        if ret:
            cv2.imwrite(self.ruta_imagen, imagen)
            self.mostrar_imagen_capturada()
            self.boton_procesar.config(state='normal')
            self.texto_resultados.config(state='normal')
            self.texto_resultados.delete(1.0, tk.END)
            self.texto_resultados.insert(tk.END, "Imagen capturada.\nHaz clic en 'Procesar Imagen' para analizar.\n")
            self.texto_resultados.config(state='disabled')
        else:
            messagebox.showerror("Error", "No se pudo capturar la imagen.")

    def mostrar_imagen_capturada(self):
        img = Image.open(self.ruta_imagen)
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.label_imagen.config(image=img_tk)
        self.label_imagen.image = img_tk

    def procesar_imagen(self):
        try:
            persona_detectada, objetos = self.detectar_personas_y_objetos()
            self.texto_resultados.config(state='normal')
            self.texto_resultados.delete(1.0, tk.END)
            self.texto_resultados.insert(tk.END, "Resultados:\n")
            self.texto_resultados.insert(tk.END, f"Persona detectada: {'Sí' if persona_detectada else 'No'}\n")
            if objetos:
                self.texto_resultados.insert(tk.END, "\nOtros objetos:\n")
                for nombre, confianza in objetos:
                    self.texto_resultados.insert(tk.END, f"- {nombre} (confianza: {confianza:.2f})\n")
            self.texto_resultados.config(state='disabled')

            # Borrar imagen
            if os.path.exists(self.ruta_imagen):
                os.remove(self.ruta_imagen)
                self.label_imagen.config(image='')
                self.boton_procesar.config(state='disabled')
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar: {e}")

    def detectar_personas_y_objetos(self):
        canal = ClarifaiChannel.get_grpc_channel()
        cliente = service_pb2_grpc.V2Stub(canal)
        metadatos = (('authorization', f'Key {self.CLARIFAI_API_KEY}'),)

        with open(self.ruta_imagen, "rb") as f:
            imagen_data = f.read()

        solicitud = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(app_id=self.CLARIFAI_APP_ID),
            model_id="aaa03c23b3724a16a56b629203edc62c",  # Modelo General
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=imagen_data
                        )
                    )
                )
            ]
        )

        respuesta = cliente.PostModelOutputs(solicitud, metadata=metadatos)

        if respuesta.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Error en la API: {respuesta.status.description}")

        conceptos = respuesta.outputs[0].data.concepts
        persona_detectada = False
        objetos = []

        # Buscar conceptos relacionados con personas
        for concepto in conceptos:
            nombre = concepto.name.lower()
            confianza = concepto.value
            if nombre in ["person", "adult", "man", "woman", "portrait"] and confianza > 0.9:
                persona_detectada = True
                break
            objetos.append((concepto.name, confianza))

        self.texto_resultados.config(state='normal')
        self.texto_resultados.insert(tk.END, f"Confianza para 'person': {next((c.value for c in conceptos if c.name.lower() == 'person'), 0.0):.2f}\n")
        for nombre, confianza in [(c.name, c.value) for c in conceptos]:
            self.texto_resultados.insert(tk.END, f"- {nombre} (confianza: {confianza:.2f})\n")
        self.texto_resultados.config(state='disabled')

        return persona_detectada, objetos

    def __del__(self):
        # Liberar cámara al cerrar
        if hasattr(self, 'camara') and self.camara.isOpened():
            self.camara.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = DeteccionPersonasApp(root)
    root.mainloop()
