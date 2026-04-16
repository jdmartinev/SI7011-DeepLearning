# Taller – NLP con Transformers (TweetEval)

Este ejercicio está dividido en varias partes y tiene como objetivo construir un pipeline completo de **procesamiento de lenguaje natural (NLP)** utilizando modelos Transformer.

## Estructura

### Parte 1 – Procesamiento de datos
Notebook: `tweeteval-part-1-data.ipynb`

En esta sección se realiza:
- Carga del dataset TweetEval
- Exploración de los datos
- Limpieza y preprocesamiento de texto

El objetivo es preparar los datos para su uso en modelos de NLP.

### Parte 2 – Pipeline base
Notebook: `tweeteval-part-2-pipeline.ipynb`

En esta sección se implementa:
- Tokenización
- Construcción de dataloaders
- Pipeline básico de entrenamiento

Se establece una base reproducible para experimentación.

### Parte 3 – DistilBERT
Notebook: `tweeteval-part-3-distilbert.ipynb`

Entrenamiento y evaluación de un modelo **DistilBERT**, una versión ligera de BERT.  
Se analiza su desempeño como baseline basado en transformers.

### Parte 4 – BERTweet
Notebook: `tweeteval-part-4-bertweet.ipynb`

Uso de **BERTweet**, un modelo especializado en texto de Twitter.  
Se comparan resultados con modelos generales.

### Parte 5 – LoRA (Fine-tuning eficiente)
Notebook: `tweeteval-part-5-lora.ipynb`

Aplicación de **LoRA (Low-Rank Adaptation)** para fine-tuning eficiente de modelos grandes.  
Se busca reducir costo computacional manteniendo desempeño.

### Parte 6 – Despliegue
Notebook: `tweeteval-part-6-deployment.ipynb`

Implementación de:
- Inferencia del modelo
- Preparación para uso en producción

Se cierra el pipeline llevando el modelo a un entorno aplicable.

## Objetivo general

Construir un pipeline completo de NLP basado en Transformers, desde el procesamiento de datos hasta el despliegue de modelos, explorando diferentes estrategias de fine-tuning.
