package org.carrot2.algorithm.optics;

import java.util.*;
import java.math.*;

import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.list.DoubleArrayList;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;
import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix1D;
import org.carrot2.core.*;
import org.carrot2.core.attribute.*;
import org.carrot2.util.IntArrayPredicateIteratorTest;
import org.carrot2.util.Pair;
import org.carrot2.util.PriorityQueue;
import org.carrot2.util.attribute.*;
import org.carrot2.util.attribute.constraint.IntRange;
import org.carrot2.util.attribute.constraint.NotBlank;
import org.carrot2.util.attribute.constraint.DoubleRange;
import org.carrot2.text.analysis.ITokenizer;
import org.carrot2.text.clustering.IMonolingualClusteringAlgorithm;
import org.carrot2.text.clustering.MultilingualClustering;
import org.carrot2.text.preprocessing.LabelFormatter;
import org.carrot2.text.preprocessing.PreprocessingContext;
import org.carrot2.text.preprocessing.pipeline.BasicPreprocessingPipeline;
import org.carrot2.text.preprocessing.pipeline.IPreprocessingPipeline;
import org.carrot2.text.vsm.TermDocumentMatrixBuilder;
import org.carrot2.text.vsm.TermDocumentMatrixReducer;
import org.carrot2.text.vsm.VectorSpaceModelContext;

import com.carrotsearch.hppc.IntArrayList;
import com.carrotsearch.hppc.IntIntOpenHashMap;
import com.carrotsearch.hppc.cursors.IntCursor;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.sorting.IndirectComparator;
import com.carrotsearch.hppc.sorting.IndirectSort;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

/**
 * Clusters documents into a flat structure based on the values of some field of the
 * documents. By default the {@link Document#SOURCES} field is used.
 */
@Bindable(prefix = "OpticsAlgorithm", inherit = CommonAttributes.class)
@Label("By OPTICS Algorithm")
public class OpticsAlgorithm extends ProcessingComponentBase implements
    IClusteringAlgorithm
{
    /**
     * Documents to cluster.
     */
    @Processing
    @Input
    @Internal
    @Attribute(key = AttributeNames.DOCUMENTS, inherit = true)
    public List<Document> documents;

    /**
     * Clusters created by the algorithm.
     */
    @Processing
    @Output
    @Internal
    @Attribute(key = AttributeNames.CLUSTERS, inherit = true)
    public List<Cluster> clusters = null;

    
    @Processing
    @Input
    @Attribute
    @IntRange(min = 2)
    @Group("Fields")
    @Level(AttributeLevel.BASIC)
    @Label("MinPts")
    public int minPts = 5;
    
    @Input
    @Processing
    @Attribute
    @DoubleRange(min = 0.0, max = 60.00)
    @Group("Fields")
    @Level(AttributeLevel.BASIC)
    @Label("Eps")    
    public double eps = 1.5;
    
    /**
     * Label count. The minimum number of labels to return for each cluster.
     */
    @Processing
    @Input
    @Attribute
    @IntRange(min = 1, max = 10)
    @Group(DefaultGroups.CLUSTERS)
    @Level(AttributeLevel.BASIC)
    @Label("Label count")
    public int labelCount = 3;
    
    public IPreprocessingPipeline preprocessingPipeline = new BasicPreprocessingPipeline();

    /**
     * Term-document matrix builder for the algorithm, contains bindable attributes.
     */
    public final TermDocumentMatrixBuilder matrixBuilder = new TermDocumentMatrixBuilder();

    /**
     * Term-document matrix reducer for the algorithm, contains bindable attributes.
     */
    public final TermDocumentMatrixReducer matrixReducer = new TermDocumentMatrixReducer();

    /**
     * Cluster label formatter, contains bindable attributes.
     */
    public final LabelFormatter labelFormatter = new LabelFormatter();

    /**
     * A helper for performing multilingual clustering.
     */
    public final MultilingualClustering multilingualClustering = new MultilingualClustering();
    
    
    /**
     * Performs by URL clustering.
     */
    @Override
    public void process() throws ProcessingException
    {
    	final List<Document> originalDocuments = documents;
        clusters = multilingualClustering.process(documents,
            new IMonolingualClusteringAlgorithm()
            {
                public List<Cluster> process(List<Document> documents,
                    LanguageCode language)
                {
                    OpticsAlgorithm.this.documents = documents;
                    OpticsAlgorithm.this.cluster(language);
                    return OpticsAlgorithm.this.clusters;
                }
            });
        documents = originalDocuments;
    }

	private void cluster(LanguageCode language)
    {
    	final PreprocessingContext preprocessingContext = 
               preprocessingPipeline.preprocess(documents, null, language);
        // Add trivial AllLabels so that we can reuse the common TD matrix builder
        final int [] stemsMfow = preprocessingContext.allStems.mostFrequentOriginalWordIndex;
        final short [] wordsType = preprocessingContext.allWords.type;
        final IntArrayList featureIndices = new IntArrayList(stemsMfow.length);
        for (int i = 0; i < stemsMfow.length; i++)
        {
            final short flag = wordsType[stemsMfow[i]];
            if ((flag & (ITokenizer.TF_COMMON_WORD | ITokenizer.TF_QUERY_WORD | ITokenizer.TT_NUMERIC)) == 0)
            {
                featureIndices.add(stemsMfow[i]);
            }
        }
        preprocessingContext.allLabels.featureIndex = featureIndices.toArray();
        preprocessingContext.allLabels.firstPhraseIndex = -1;
        // Further processing only if there are words to process
        clusters = Lists.newArrayList();
        
        final VectorSpaceModelContext vsmContext = new VectorSpaceModelContext(
                preprocessingContext);
        matrixBuilder.buildTermDocumentMatrix(vsmContext);
        
        final IntIntOpenHashMap rowToStemIndex = new IntIntOpenHashMap();
        for (IntIntCursor c : vsmContext.stemToRowIndex)
        {
            rowToStemIndex.put(c.value, c.key);
        }
        final DoubleMatrix2D tdMatrix;
        tdMatrix = vsmContext.termDocumentMatrix;
        
        final List<IntArrayList> rawClusters = Lists.newArrayList();
        TreeSet<Integer> unvisited = new TreeSet<Integer>();
        DoubleArrayList reachDistArray = new DoubleArrayList();
        IntArrayList neighborsOuter;
        IntArrayList neighborsInner;
        IntArrayList orderedList = new IntArrayList();
//      tu jeszcze comparator, pewnie na podstawie odleglosci w drugim elemencie pary
        java.util.PriorityQueue<Pair<Integer, Double>> seeds = 
        		new java.util.PriorityQueue<Pair<Integer, Double>>(tdMatrix.columns(), 
        				new Comparator<Pair<Integer, Double>>() {
							@Override
							public int compare(Pair<Integer, Double> o1,
									Pair<Integer, Double> o2) {
								if (o1.objectB > o2.objectB){
									return -1;
								} else if (o1.objectB < o2.objectB){
									return 1;
								} else{
									return 0;
								}
							}
						});
        //OPTICS
        for (int i=0; i<tdMatrix.columns(); i++){
        	reachDistArray.add(-1);	// UNDEFINED jako reachability-distance dla każdego dokumentu
        	unvisited.add(i);
        }
        while (!unvisited.isEmpty()) {
        	int p = unvisited.first();
        	neighborsOuter = getNeighbors(p, tdMatrix);
        	unvisited.remove(p);	// as processed
        	orderedList.add(p);
        	if (coreDistance(p, tdMatrix) != -1){
        		update(tdMatrix, neighborsOuter, p, seeds, unvisited, reachDistArray);
//        		Iterator<Pair<Integer, Double>> seedsIter = seeds.iterator();
//        		while(seedsIter.hasNext()){
        		for (Pair<Integer, Double> pair : seeds) {
        			Pair<Integer, Double> queueElem = pair;
        			int q = queueElem.objectA.intValue();
        			neighborsInner = getNeighbors(q, tdMatrix);
        			unvisited.remove(q);
        			orderedList.add(q);
        			if (coreDistance(q, tdMatrix) != -1){
        				java.util.PriorityQueue<Pair<Integer, Double>> copySeeds = new java.util.PriorityQueue<Pair<Integer, Double>>(tdMatrix.columns(), 
                				new Comparator<Pair<Integer, Double>>() {
        							@Override
        							public int compare(Pair<Integer, Double> o1,
        									Pair<Integer, Double> o2) {
        								if (o1.objectB > o2.objectB){
        									return -1;
        								} else if (o1.objectB < o2.objectB){
        									return 1;
        								} else{
        									return 0;
        								}
        							}
        						});
        				copySeeds.addAll(seeds);
        				update(tdMatrix, neighborsInner, q, copySeeds, unvisited, reachDistArray);
        			}
        		}
        	}
        }
        makeClusters(orderedList, eps/3, tdMatrix, rawClusters, reachDistArray);
        
        for (int i = 0; i < rawClusters.size(); i++){
            final Cluster cluster = new Cluster();
            final IntArrayList rawCluster = rawClusters.get(i);
            if (rawCluster.size() > 1){
                cluster.addPhrases(getLabels(rawCluster,
                    vsmContext.termDocumentMatrix, rowToStemIndex,
                    preprocessingContext.allStems.mostFrequentOriginalWordIndex,
                    preprocessingContext.allWords.image));
                for (int j = 0; j < rawCluster.size(); j++){
                    cluster.addDocuments(documents.get(rawCluster.get(j)));
                }
                clusters.add(cluster);
            }
        }

    Collections.sort(clusters, Cluster.BY_REVERSED_SIZE_AND_LABEL_COMPARATOR);
    Cluster.appendOtherTopics(documents, clusters);
        
    }
    
    private List<String> getLabels(IntArrayList documents,
            DoubleMatrix2D termDocumentMatrix, IntIntOpenHashMap rowToStemIndex,
            int [] mostFrequentOriginalWordIndex, char [][] wordImage)
    {
        final DoubleMatrix1D words = new DenseDoubleMatrix1D(termDocumentMatrix.rows());
        for (IntCursor d : documents)
        {
            words.assign(termDocumentMatrix.viewColumn(d.value), Functions.PLUS);
        }
        final List<String> labels = Lists.newArrayListWithCapacity(labelCount);
        final int [] order = IndirectSort.mergesort(0, words.size(),
            new IndirectComparator()
            {
                @Override
                public int compare(int a, int b)
                {
                    final double valueA = words.get(a);
                    final double valueB = words.get(b);
                    return valueA < valueB ? -1 : valueA > valueB ? 1 : 0;
                }
            });
        final double minValueForLabel = words.get(order[order.length
            - Math.min(labelCount, order.length)]);
        for (int i = 0; i < words.size(); i++)
        {
            if (words.getQuick(i) >= minValueForLabel)
            {
                labels.add(LabelFormatter.format(new char [] []
                {
                    wordImage[mostFrequentOriginalWordIndex[rowToStemIndex.get(i)]]
                }, new boolean []
                {
                    false
                }, false));
            }
        }
        return labels;
    }
    
    private double getDistance(DoubleMatrix1D a, DoubleMatrix1D b) {
    	double dist=0;
    	for (int i=0; i< a.size(); ++i) {
    		dist += Math.sqrt( Math.pow((a.get(i)-b.get(i)), 2));
    	}
    	return dist;
    }
    
    private IntArrayList getNeighbors(int p, DoubleMatrix2D tdMatrix) {
    	IntArrayList neighbors = new IntArrayList();
    	for(int i = 0; i<tdMatrix.columns(); ++i) {
    		if(i==p){
    			neighbors.add(p);
    		} else if(getDistance(tdMatrix.viewColumn(i), tdMatrix.viewColumn(p)) <= eps){
    			neighbors.add(i);
    		}
    	}
    	return neighbors;
    }
    
    private double coreDistance(int p, DoubleMatrix2D tdMatrix){
    	double dist = 0;
    	double actualDist = 0;
    	IntArrayList neighborsP = getNeighbors(p, tdMatrix);
    	if (neighborsP.size() < minPts){
    		dist = -1; //UNDEFINED
    	} else {
    		for (int i=0; i<neighborsP.size(); i++){
    			actualDist = getDistance(tdMatrix.viewColumn(neighborsP.get(i)), tdMatrix.viewColumn(p));
    			if (dist == 0 || (actualDist < dist)){
    				dist = actualDist;
    			}
    		}
    	}
    	return dist;
    }
    
    private double reachabilityDistance(int p, int o, DoubleMatrix2D tdMatrix){
    	double dist = 0;
    	IntArrayList neighborsP = getNeighbors(p, tdMatrix);
    	if (neighborsP.size() < minPts){
    		dist = -1;
    	} else {
    		double coreDist = coreDistance(p, tdMatrix);
    		double euclidesDist = getDistance(tdMatrix.viewColumn(p), tdMatrix.viewColumn(o));
    		dist = coreDist > euclidesDist ? coreDist : euclidesDist;
    	}
    	return dist;
    }
    
    private void update(DoubleMatrix2D tdMatrix, IntArrayList N ,int p, 
    		java.util.PriorityQueue<Pair<Integer, Double>> seeds, TreeSet<Integer> unvisited,
    		DoubleArrayList reachDistArray){
    	double newReachDist = 0;
    	// N.get(i) jako o w algorytmie
    	for (int i=0; i<N.size(); i++){
    		if (!unvisited.contains(N.get(i))){
    			double coreDist = coreDistance(p, tdMatrix);
        		double euclidesDist = getDistance(tdMatrix.viewColumn(p), tdMatrix.viewColumn(N.get(i)));
        		newReachDist = coreDist > euclidesDist ? coreDist : euclidesDist;
        		if (reachDistArray.get(N.get(i)) == -1){	//UNDEFINED
        			reachDistArray.set(N.get(i), newReachDist);
        			seeds.add(new Pair<Integer, Double>(N.get(i), newReachDist));
        		} else {
        			if (newReachDist < reachDistArray.get(N.get(i))){
        				//update seeds jako usuniecie starego elementu, zmiana i dodanie nowego
        				Pair<Integer, Double> oldPair = new Pair<Integer, Double>(N.get(i), reachDistArray.get(N.get(i)));
        				reachDistArray.set(N.get(i), newReachDist);
        				seeds.remove(oldPair);
        				Pair<Integer, Double> newPair = new Pair<Integer, Double>(N.get(i), newReachDist);
        				seeds.add(newPair);
        			}
        		}
    		}
    	}
    }
    
    void makeClusters(IntArrayList orderedList, double Ei, DoubleMatrix2D tdMatrix, 
    		List<IntArrayList> rawClusters, DoubleArrayList reachDistArray){
    	IntArrayList szumCluster = new IntArrayList();
    	IntArrayList cluster = new IntArrayList();
    	for (IntCursor p : orderedList){
			if (reachDistArray.get(p.value) > Ei){
				if (coreDistance(p.value, tdMatrix) <= Ei){
					if (cluster.size() != 0){
						rawClusters.add(cluster);
					}
					cluster = new IntArrayList();
					cluster.add(p.value);
				} else {
					szumCluster.add(p.value);
				}
			} else {
				cluster.add(p.value);
			}
		}
    	rawClusters.add(szumCluster);
    	if (!rawClusters.contains(cluster)){
    		rawClusters.add(cluster);
    	}
    }
    
}
