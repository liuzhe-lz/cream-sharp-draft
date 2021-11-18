namespace Cream {
    public class ConvBnAct : DynamicModule
    {
        public Module conv = null;
        public Module bn1 = null;
        public Module act1 = null;

        public bool has_residual = false;
        public double drop_path_rate = 0;

        public ConvBnAct(string name) : base(name) { }

        public override Tensor forward(Tensor x)
        {
            Tensor shortcut = x;
            x = conv.forward(x);
            x = bn1.forward(x);
            x = act1.forward(x);
            x = Utils.HandleResidual(has_residual, drop_path_rate, shortcut, x);
            return x;
        }
    }
}
