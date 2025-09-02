# --- Language dictionary ---
texts = {
    "title": {
        "English": "Surf Detection with AI",
        "Español": " Detección de Surf con IA"
    },
    "algo_header": {
        "English": "Algorithm Explanation",
        "Español": "¿Para qué un algoritmo de detección de surf?"
    },
    "algo_desc": {
        "English": (
            "Surfing is my passion. To progress in this sport, it is key to watch yourself surfing and be able to identify mistakes and successes.\n\n"
            "In many beaches that are surf spots, there are cameras constantly filming. Generally, these are used so surfers can check the conditions and decide whether to go to that spot or look for another.\n\n"
            "But in surfing, most of the time you are paddling or waiting for a wave; out of a 1-hour session, only a few minutes are actually spent standing on the board surfing.\n\n"
            "This algorithm aims to detect the moments when surfers are actually riding waves.\n\n"
            "As a corollary, after detection, the application generates a highlight video summary of the session, showing only the surfed waves."
        ),
        "Español": (
            "El surf es mi pasión. Para progresar en este deporte es clave verse a uno mismo surfiando y poder identificar errores y aciertos. \n\n"
            "En muchísimas playas que son surf points, existen cámaras filmando constantemente. En general se usan para que el surfista pueda mirar cómo está el point y decidir si ir a ese o buscar otro point.\n\n"
            "Pero en el surf la mayor parte del tiempo uno está nadando o esperando la ola, del total del tiempo de una sesión de 1 hora, son pocos minutos los que uno está parado arriba de la tabla surfiando.\n\n"
            "Este algoritmo lo que busca es poder detectar los momentos en donde hay surfistas efectivamente montados surfiando la ola.\n\n"
            "Cómo corolario, luego de la detección, la aplicación incorpora la generación de un video resumen de los highlights de la sesión donde se ven solo las olas surfiadas. "
        )
    },
"training_header": {
    "English": "How was the model trained?",
    "Español": "¿Cómo se entrenó el modelo de detección?"
},
"training_desc": {
    "English": (
        "To train the detection model, I obtained videos from Playa Grande, Mar del Plata. "
        "Frames were extracted from these videos every 2 seconds to create a diverse set of images.\n\n"
        "This labeling is fundamental for the model to learn the difference between 'surfing' and 'not surfing'.\n\n"
        "With the Roboflow tool, images of surfers riding waves and also surfers not surfing were manually labeled, to clearly distinguish both situations.\n\n"
        'Through Roboflow we also generated video "augmentation" to create artificial variations of the images (rotations, lighting changes, cropping, etc.) and thus increase the size and diversity of the dataset to improve the model\'s generalization.\n\n'
        "Finally, to train the model, I used the YOLO (You Only Look Once) architecture, which is widely used for real-time object detection."

    ),
    "Español": (
        "Para entrenar el modelo de detección, conseguí videos de Playa Grande, Mar del Plata. "
        "Se extrajeron frames de estos videos cada 2 segundos para crear un conjunto diverso de imágenes.\n\n"
        "Este etiquetado es fundamental para que el modelo aprenda la diferencia entre 'surfeando' y 'no surfeando'.\n\n"
        "Con la herramienta Roboflow se etiquetaron manualmente imágenes de surfistas surfeando olas y también de surfistas que no están surfeando, para distinguir claramente ambas situaciones.\n\n"
        'A través de Roboflow también generamos "augmentation" del video para generar variaciones artificiales de las imágenes (rotadaciones, con cambios de luz, recortes, etc.) y así aumentar el tamaño y la diversidad del dataset para mejorar la generalización del modelo.\n\n'
        "Finalmente para entrenar el modelo, usé la arquitectura YOLO (You Only Look Once), ampliamente utilizada para detección de objetos en tiempo real."
    )
},

    "demo_header": {
        "English": "Demo Video",
        "Español": "Video Demo"
    },
    "demo_not_found": {
        "English": "Demo video not found.",
        "Español": "Video de demostración no encontrado."
    },
    "upload_header": {
        "English": "Upload Your Video for Detection",
        "Español": "Subí tu Video para correr una Detección"
    },
    "upload_prompt": {
        "English": "Choose a video...",
        "Español": "Elegí un video..."
    },
    "running_detection": {
        "English": "Running surf detection on your video...",
        "Español": "Corriendo la detección de surf en tu video..."
    },
    "error_processing": {
        "English": "Error processing the video.",
        "Español": "Error procesando el video."
    }
}