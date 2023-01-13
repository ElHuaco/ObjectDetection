# Asumimos que la prediccion es una lista de tensores (c + 4) x DBs x H x W para cada escala
#   Para cada escala: (n_cats + 4) * n_default_boxes * H_scale * W_scale
# La pérdida entonces es, para cada ground truth box de la imagen:
#   i) Las predicciones traducidas a relative-space desde su escala, se matchean con Jaccard a cada ground truth
#   esto se hace con Matching.py
#   ii) Para las que tengan jaccard > 0.5, se calcula L1 de Offset y Softmax de Categorías. Aquí.

