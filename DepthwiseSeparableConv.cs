namespace Cream {
    public class DepthwiseSeparableConv : DynamicModule
    {
        public Module conv_dw = null;
        public Module bn1 = null;
        public Module act1 = null;
        public Module se = null;
        public Module conv_pw = null;
        public Module bn2 = null;
        public Module act2 = null;

        public bool has_residual = false;
        public double drop_path_rate = 0;

        public DepthwiseSeparableConv(string name) : base(name) { }

        public override Tensor forward(Tensor x)
        {
            Tensor shortcut = x;

            x = conv_dw.forward(x);
            x = bn1.forward(x);
            x = act1.forward(x);

            x = se.forward(x);

            x = conv_pw.forward(x);
            x = bn2.forward(x);
            x = act2.forward(x);

            x = Utils.HandleResidual(has_residual, drop_path_rate, shortcut, x);
            return x;
        }
    }
}
