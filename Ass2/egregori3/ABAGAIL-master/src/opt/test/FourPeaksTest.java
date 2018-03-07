package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 20;
    /** The t value */
    private static final int T = N / 10;
    
    public static void main(String[] args) 
    {
        int correctRHC = 0;
        int correctSA = 0;
        int correctGA = 0;
        int correctMIMIC = 0;
        for(int i=0; i<1000; ++i)
        {
            int[] ranges = new int[N];
            System.out.println("ranges: " + ranges);
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new FourPeaksEvaluationFunction(T);   // eval the evaulation function
            Distribution odd = new DiscreteUniformDistribution(ranges);   // dist the initial distribution
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);  // neigh the neighbor function
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges); 

            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 1000);
            fit.train();
// System.out.println("RHC: " + ef.value(rhc.getOptimal()));

            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
            fit = new FixedIterationTrainer(sa, 1000);
            fit.train();
//        System.out.println("SA: " + ef.value(sa.getOptimal()));
//        
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(N, N/2, N/10, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
//        System.out.println("GA: " + ef.value(ga.getOptimal()));
//        
            MIMIC mimic = new MIMIC(N, N/10, pop);
            fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
//        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));

            if(ef.value(rhc.getOptimal()) == 37 )
                correctRHC += 1;
            if(ef.value(sa.getOptimal()) == 37 )
                correctSA += 1;
            if(ef.value(ga.getOptimal()) == 37 )
                correctGA += 1;
            if(ef.value(mimic.getOptimal()) == 37 )
                correctMIMIC += 1;

        }
        System.out.println("correctRHC: " + correctRHC);
        System.out.println("correctSA: " + correctSA);
        System.out.println("correctGA: " + correctGA);
        System.out.println("correctMIMIC: " + correctMIMIC);
    }
}
