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
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * @author Daniel Cohen dcohen@gatech.edu
 * @version 1.0
 */
public class TwoColorsTest {
    /** The number of colors */
    private static final int k = 2;
    /** The N value */
    private static final int N = 100*k;

    public static void main(String[] args) 
    {
        int correctRHC = 0;
        int correctSA = 0;
        int correctGA = 0;
        int correctMIMIC = 0;
        double mymax = 0.0;
        for(int i=0; i<1000; ++i)
        {
            int[] ranges = new int[N];
            Arrays.fill(ranges, k+1);
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
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 100);
            fit.train();
//            System.out.println(ef.value(rhc.getOptimal()));
            
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, 100);
            fit.train();
//            System.out.println(ef.value(sa.getOptimal()));
            
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
            fit = new FixedIterationTrainer(ga, 100);
            fit.train();
//            System.out.println(ef.value(ga.getOptimal()));
            
            MIMIC mimic = new MIMIC(50, 10, pop);
            fit = new FixedIterationTrainer(mimic, 100);
            fit.train();
//            System.out.println(ef.value(mimic.getOptimal()));


            double rhcv = ef.value(rhc.getOptimal());
            double sav = ef.value(sa.getOptimal());
            double gav = ef.value(ga.getOptimal());
            double mimicv = ef.value(mimic.getOptimal());

            if( rhcv== N )
                correctRHC += 1;
            if( sav== N )
                correctSA += 1;
            if( gav== N )
                correctGA += 1;
            if( mimicv== N )
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
