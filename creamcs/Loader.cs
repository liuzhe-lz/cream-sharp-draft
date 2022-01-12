using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

class Loader
{
    static public Module LoadArchAndWeights(string path) {
        var archJson = File.ReadAllText(Path.Join(path, "architecture.json"));
        var arch = JsonConvert.DeserializeObject(archJson)!;
        var model = LoadArch(arch);

        LoadWeights(model, path);

        return model;
    }

    static public Module LoadArch(dynamic arch)
    {
        if (arch.type == "AdaptiveAvgPool2d") {
            var outputSize = (long)arch.kwargs.output_size;
            return AdaptiveAvgPool2d(new long[] { outputSize, outputSize });
        }
    
        if (arch.type == "BatchNorm2d") {
            return BatchNorm2d(
                (long)arch.args[0],
                (double)arch.kwargs.eps,
                (double)arch.kwargs.momentum,
                (bool)arch.kwargs.affine,
                (bool)arch.kwargs.track_running_stats
            );
        }
    
        if (arch.type == "Conv2d") {
            return Conv2d(
                (long)arch.args[0],
                (long)arch.args[1],
                (long)arch.kwargs.kernel_size,
                (long)arch.kwargs.padding,
                (long)arch.kwargs.stride,
                (long)arch.kwargs.dilation,
                TorchSharp.PaddingModes.Zeros,
                (long)arch.kwargs.groups,
                (bool)arch.kwargs.bias
            );
        }
    
        if (arch.type == "Identity") {
            return Identity();
        }
    
        if (arch.type == "Linear") {
            return Linear(
                (long)arch.kwargs.in_features,
                (long)arch.kwargs.out_features,
                (bool)arch.kwargs.bias
            );
        }
    
        if (arch.type == "ReLU") {
            return ReLU((bool)arch.kwargs.inplace);
        }
    
        if (arch.type == "Sequential") {
            return new SequentialWrapper((string)arch.name, arch.children);
        }
    
        var module_class = Type.GetType((string)arch.type)!;
        var ctor = module_class.GetConstructor(new [] { typeof(string), typeof(object) })!;
        return (Module)ctor.Invoke(new object[] { (string)arch.name, arch.children });
    }

    static public void LoadWeights(Module model, string path)
    {
        var metaJson = File.ReadAllText(Path.Join(path, "weights_meta.json"));
        var arr = JsonConvert.DeserializeObject<JArray>(metaJson);
        foreach (dynamic obj in arr!) {
            var shape = ((JArray)obj.shape).Select(x => (long)x).ToArray();
            var tensor = Loader.ReadTensor(Path.Join(path, "weights", obj.name), shape, obj.dtype);
            AssignWeight(model, obj.name, tensor);
        }
    }

    static public Tensor ReadTensor(string path, long[] shape, string dtypeName)
    {
        long n = shape.Aggregate((long)1, (x, y) => x * y);
        var bytes = File.ReadAllBytes(path);
        Tensor? tensor = null;
    
        if (dtypeName == "torch.float32") {
            var data = new float[n];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            tensor = torch.tensor(data, torch.float32);
    
        } else if (dtypeName == "torch.float64") {
            var data = new double[n];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            tensor = torch.tensor(data, torch.float32);
    
        } else if (dtypeName == "torch.uint8") {
            tensor = torch.tensor(bytes, torch.uint8);
    
        } else if (dtypeName == "torch.int8") {
            var data = new sbyte[n];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            tensor = torch.tensor(data, torch.int8);
    
        } else if (dtypeName == "torch.int16") {
            var data = new short[n];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            tensor = torch.tensor(data, torch.int16);
    
        } else if (dtypeName == "torch.int32") {
            var data = new int[n];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            tensor = torch.tensor(data, torch.int32);
    
        } else if (dtypeName == "torch.int64") {
            var data = new long[n];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            tensor = torch.tensor(data, torch.int64);
    
        } else if (dtypeName == "torch.bool") {
            var data = new bool[n];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            tensor = torch.tensor(data, torch.@bool);
        }
    
        return tensor!.reshape(shape);
    }

    static public void AssignWeight(Module model, string layerName, Tensor tensor)
    {
        object current = model;
        var parts = layerName.Split('.');
        foreach (var part in parts.SkipLast(1)) {
            int index;
            var isNum = int.TryParse(part, out index);
            if (isNum) {
                current = current.GetType()!.GetProperty("Item")!.GetValue(current, new object[] { index })!;
            } else {
                var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
                current = current.GetType()!.GetField(part, flags)!.GetValue(current)!;
            }
        }
        current.GetType()!.GetProperty(parts.Last())!.SetValue(current, tensor);
    }
}
