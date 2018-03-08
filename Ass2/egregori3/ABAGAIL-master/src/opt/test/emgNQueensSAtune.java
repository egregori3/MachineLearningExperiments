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

import java.util.Random;
import opt.ga.NQueensFitnessFunction;
import dist.DiscretePermutationDistribution;
import opt.SwapNeighbor;
import opt.ga.SwapMutation;
/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class emgNQueensSAtune 
{
    private static void unitTest(int N, int optima, double rate)
    {
        int runs = 1000;
        double min = (double)runs;
        double max = 0.0;
        double sum = 0.0;
        System.out.print(optima+",");
        int successes = 0;
        for( int a=0; a<runs; a++ )
        {
            int[] ranges = new int[N];
            Random random = new Random(N);
            for (int i = 0; i < N; i++) {
                ranges[i] = random.nextInt();
            }
            NQueensFitnessFunction ef = new NQueensFitnessFunction();
            NeighborFunction nf = new SwapNeighbor();
            Distribution odd = new DiscretePermutationDistribution(N);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);     
            SimulatedAnnealing sa = new SimulatedAnnealing(1E1, rate, hcp);

            emgEqualityTrainer fit = new emgEqualityTrainer(sa, optima, runs);
            double result = fit.train();
            if( result >= 0 )
            {
                successes += 1;
                sum += result;
                if( result < min ) min = result;
                if( result > max ) max = result;
            }
        }
        System.out.println(successes+","+(sum/(double)successes));
    }

    public static void main(String[] args) 
    {
        int[] optima = {45,190,435,778,1221,1763,2405};
        double[] rate = {0.05, 0.5, 0.95, 0.99};
        int N = 30;
        int P = optima[(N/10)-1];
        for( int i=0; i<4; ++i )
        {
            System.out.print(N+","+rate[i]+"->");
            unitTest(N, P, rate[i]);
        }
        N = 40;
        P = optima[(N/10)-1];
        for( int i=0; i<4; ++i )
        {
            System.out.print(N+","+rate[i]+"->");
            unitTest(N, P, rate[i]);
        }

    }
}
