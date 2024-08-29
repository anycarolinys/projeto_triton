### **Projeto de Deploy de Modelos com Triton Server**


Reprodução do deploy de um modelo de Machine Learning no Triton Server utilizando Podman.
A ideia é serializar um modelo de regressão linear para o formato .pickle, subir este arquivo num servidor Triton e poder realizar inferências com o modelo utilizando requests HTTP. 