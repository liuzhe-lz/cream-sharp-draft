var model = Loader.LoadArchAndWeights("export");
model.save("weights.dat");

model = Loader.LoadArch("export");
model.load("weights.dat");
