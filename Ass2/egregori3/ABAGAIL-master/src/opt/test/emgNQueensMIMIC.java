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
public class emgNQueensMIMIC 
{
    private static void unitTest(int N, int optima)
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
            Distribution odd = new DiscretePermutationDistribution(N);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1); 
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);     
            SimulatedAnnealing sa = new SimulatedAnnealing(1E1, .1, hcp);
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 0, 10, gap);
            MIMIC mimic = new MIMIC(200, 10, pop);

            emgEqualityTrainer fit = new emgEqualityTrainer(mimic, optima, runs);
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
        int[] optima = {45,190,435,778,1221,1763,2405};
        for( int N=10; N<=40; N+=10 )
        {
            System.out.print(N+",");
            unitTest(N, optima[(N/10)-1]);
        }
    }
}
