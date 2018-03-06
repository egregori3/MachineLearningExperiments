package shared;

/**
 * A fixed iteration trainer
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class emgEqualityTrainer implements Trainer {
    
    /**
     * The inner trainer
     */
    private Trainer trainer;
    
    /**
     * The number of iterations to train
     */
    private int desiredvalue;
    private int maxiteration;

    /**
     * Make a new fixed iterations trainer
     * @param t the trainer
     * @param iter the number of iterations
     */
    public emgEqualityTrainer(Trainer t, int dv, int mi) {
        trainer = t;
        desiredvalue = dv;
        maxiteration = mi;
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() 
    {
        int rv = 0;
        for (int i = 0; i < maxiteration; i++) 
        {
            rv = (int)trainer.train();
            if( rv == desiredvalue )
                return (double)i;
        }
        return -1.0;
    }
}
