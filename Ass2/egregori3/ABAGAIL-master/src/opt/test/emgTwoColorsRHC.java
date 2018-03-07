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
import shared.emgEqualityTrainer;
import opt.ga.UniformCrossOver;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class emgTwoColorsRHC 
{
    private static void unitTest(int N, int K)
    {
        int runs = 1000;
        double min = (double)runs;
        double max = 0.0;
        double sum = 0.0;
        int optima = N-K;
        System.out.print(optima+",");
        int successes = 0;
        for( int i=0; i<runs; i++ )
        {
            int[] ranges = new int[N];
            Arrays.fill(ranges, K+1);
            EvaluationFunction ef = new TwoColorsEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges); 
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);

            emgEqualityTrainer fit = new emgEqualityTrainer(rhc, optima, runs);
            double result = fit.train();
            if( result >= 0 )
            {
                successes += 1;
                sum += result;
                if( result < min ) min = result;
                if( result > max ) max = result;
            }
        }
        System.out.println(successes+","+min+","+(sum/(double)successes)+","+max);
    }

    public static void main(String[] args) 
    {
        int K = 2;
        for( int N=(10*K); N<=(100*K); N+=N )
        {
            System.out.print(N+","+K+",");
            unitTest(N,K);
        }
    }
}
