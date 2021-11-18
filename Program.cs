using YamlDotNet.RepresentationModel;

var input = new StreamReader("14.yaml");
var yaml = new YamlStream();
yaml.Load(input);

var root = (YamlMappingNode)yaml.Documents[0].RootNode;
var arch = (YamlMappingNode)root["ChildNet"];

var net = new CreamNetTemplate();
foreach (var entry in arch.Children) {
    Console.WriteLine(entry.Key);
    var layer = LayerBuilder.BuildLayer(entry.Value);
    Console.WriteLine(layer);
    var layerField = net.GetType().GetField((string)entry.Key);
    layerField.SetValue(net, layer);
}
