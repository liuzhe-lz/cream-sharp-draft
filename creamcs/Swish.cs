using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class Swish : Module
{
    public Swish(string name, dynamic arch) : base(name)
    {
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        return x.mul(x.sigmoid());
    }
}
