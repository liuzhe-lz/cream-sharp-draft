using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using YamlDotNet.RepresentationModel;

public class LayerBuilder
{
    /*private static Dictionary<string, bool> BuiltInLayers = new Dictionary<string, bool> {
        { "BatchNorm2d", true },
        { "Conv2d", true },
        { "Identity", true },
        { "Linear", true },
        { "ReLU", true },
    };*/

    private static Dictionary<string, Type> SimpleLayers = new Dictionary<string, Type> {
        { "SelectAdaptivePool2d", typeof(SelectAdaptivePool2d) },
        { "Swish", typeof(Swish) },
    };

    private static Dictionary<string, Type> NestedLayers = new Dictionary<string, Type> {
        { "ConvBnAct", typeof(ConvBnAct) },
        { "DepthwiseSeparableConv", typeof(DepthwiseSeparableConv) },
        { "InvertedResidual", typeof(InvertedResidual) },
        { "SqueezeExcite", typeof(SqueezeExcite) },
    };

    public static string BuildLayer(string layerName, YamlNode yaml)
    {
        if (yaml.NodeType == YamlNodeType.Sequence) {
            var seq = new List<Module>();
            for (var item in (YamlSequenceNode)yaml) {
                seq.Add(BuildLayer(item));
            }
            return Sequential(seq);
        }

        var info = (YamlMappingNode)yaml;
        var name = (string)(YamlScalarNode)info["name"];

        if (NestedLayers.ContainsKey(name)) {
            var type = NestedLayers[name];
            var obj = Activator.CreateInstance(type, layerName);
            for (var [k, v] in children) {
                var field = type.GetField(k);
                field.setValue(obj, BuildLayer(k, v));
            }
            return obj;
        }

        if (SimpleLayers.ContainsKey(name)) {
            var type = SimpleLayers[name];
            var ctor = type.GetConstructors()[0];
            var args = Utils.AlignArgs(ctor, info["args"], info["kwargs"]);
            return Activator.CreateInstance(type, args);
        } else {
            var factory = Type.GetType("TorchSharp.torch.nn." + name);
            var args = Utils.AlignArgs(factory, info["args"], info["kwargs"]);
            return method.Invoke(args);
        }
    }
}
