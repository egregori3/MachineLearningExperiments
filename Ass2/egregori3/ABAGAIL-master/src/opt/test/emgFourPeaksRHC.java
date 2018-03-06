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
public class emgFourPeaksRHC 
{
    private static void unitTest(int N, int T)
    {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);   // eval the evaulation function
        Distribution odd = new DiscreteUniformDistribution(ranges);   // dist the initial distribution
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);  // neigh the neighbor function

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        double min = 100000.0;
        double max = 0.0;
        double sum = 0.0;
        int optima = N+(N-T-1);
        System.out.print(optima+",");
        int runs = 100000;
        double results = 0.0;
        for( int i=0; i<runs; i++ )
        {
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            emgEqualityTrainer fit = new emgEqualityTrainer(rhc, optima, (int)min);
            double result = fit.train();
            if( result > 0 )
            {
                results += 1.0;
                sum += result;
                if( result < min ) min = result;
                if( result > max ) max = result;
            }
        }
        System.out.println((results/(double)runs)+","+min+","+(sum/results)+","+max);
    }

    public static void main(String[] args) 
    {
        for( int N=10; N<250; N+=N )
        {
            int T = N/10;
            System.out.print(N+","+T+",");
            unitTest(N,T);
        }
    }
}
