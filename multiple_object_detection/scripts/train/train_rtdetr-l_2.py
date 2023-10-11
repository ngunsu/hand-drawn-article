from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
model.info()
model.train(data="./data/full_train.yaml", epochs=100, project='rtdetr-l', name='2', seed=2, val=False)
