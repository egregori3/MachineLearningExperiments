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
import shared.emgEqualityTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class emgFourPeaksGA 
{
    private static void unitTest(int N, int T)
    {
        int runs = 1000;
        double min = (double)runs;
        double max = 0.0;
        double sum = 0.0;
        int optima = N+(N-T-1);
        System.out.print(optima+",");
        int successes = 0;
        for( int i=0; i<runs; i++ )
        {
            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new FourPeaksEvaluationFunction(T);   // eval the evaulation function
            Distribution odd = new DiscreteUniformDistribution(ranges);   // dist the initial distribution
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);  // neigh the neighbor function
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(N, N/2, N/10, gap);   
            emgEqualityTrainer fit = new emgEqualityTrainer(ga, optima, runs);
            double result = fit.train();
            if( result >= 0 )
            {
 //               System.out.println("GA: " + ef.value(ga.getOptimal()));
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
        for( int N=10; N<=80; N+=N )
        {
            int T = N/10;
            System.out.print(N+","+T+",");
            unitTest(N,T);
        }
    }
}
