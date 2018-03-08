package opt.test;

import java.util.Arrays;
import java.util.Random;
import opt.ga.NQueensFitnessFunction;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * @author kmanda1
 * @version 1.0
 */
public class NQueensTest {
    /** The n value */
    private static final int N = 10;
    /** The t value */
    
    public static void main(String[] args) 
    {
        int correctRHC = 0;
        int correctSA = 0;
        int correctGA = 0;
        int correctMIMIC = 0;
        double mymax = 0.0;
        for(int n=0; n<1000; ++n)
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
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 1000);
            fit.train();
            long starttime = System.currentTimeMillis();
            System.out.println("RHC: " + ef.value(rhc.getOptimal()));
            System.out.println("RHC: Board Position: ");
           // System.out.println(ef.boardPositions());
            System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
            
            System.out.println("============================");
            
            SimulatedAnnealing sa = new SimulatedAnnealing(1E1, .1, hcp);
            fit = new FixedIterationTrainer(sa, 1000);
            fit.train();
            
            starttime = System.currentTimeMillis();
            System.out.println("SA: " + ef.value(sa.getOptimal()));
            System.out.println("SA: Board Position: ");
           // System.out.println(ef.boardPositions());
            System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
            
            System.out.println("============================");
            
            starttime = System.currentTimeMillis();
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 0, 10, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            System.out.println("GA: " + ef.value(ga.getOptimal()));
            System.out.println("GA: Board Position: ");
            //System.out.println(ef.boardPositions());
            System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
            
            System.out.println("============================");
            
            starttime = System.currentTimeMillis();
            MIMIC mimic = new MIMIC(200, 10, pop);
            fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
            System.out.println("MIMIC: Board Position: ");
            //System.out.println(ef.boardPositions());
            System.out.println("Time : "+ (System.currentTimeMillis() - starttime));

            double rhcv = ef.value(rhc.getOptimal());
            double sav = ef.value(sa.getOptimal());
            double gav = ef.value(ga.getOptimal());
            double mimicv = ef.value(mimic.getOptimal());

            int optima = 45;
            if( rhcv== optima )
                correctRHC += 1;
            if( sav== optima )
                correctSA += 1;
            if( gav== optima )
                correctGA += 1;
            if( mimicv== optima )
                correctMIMIC += 1;

            if(rhcv > mymax )
                mymax = rhcv;
            if(sav > mymax )
                mymax = sav;
             if(gav > mymax )
                mymax = gav;
            if(mimicv > mymax )
                mymax = mimicv;
        }

        System.out.println("max: " + mymax);
        System.out.println("correctRHC: " + correctRHC);
        System.out.println("correctSA: " + correctSA);
        System.out.println("correctGA: " + correctGA);
        System.out.println("correctMIMIC: " + correctMIMIC);
    }
}
