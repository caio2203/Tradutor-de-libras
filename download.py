from roboflow import Roboflow

rf = Roboflow(api_key="9JBByF7hgVwfetstXQhv")
project = rf.workspace("personal-bu69s").project("libras-ih14i")
dataset = project.version(2).download("coco")  # ← Mudei aqui
