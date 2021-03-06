package org.carrot2.algorithm.dbscan;

import java.util.*;
import java.math.*;

import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;
import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix1D;
import org.carrot2.core.*;
import org.carrot2.core.attribute.*;
import org.carrot2.util.IntArrayPredicateIteratorTest;
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
@Bindable(prefix = "DBScanAlgorithm", inherit = CommonAttributes.class)
@Label("By DBScan Agorithm")
public class DBScanAlgorithm extends ProcessingComponentBase implements
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
    @DoubleRange(min = 0.0, max = 500.00)
    @Group("Fields")
    @Level(AttributeLevel.BASIC)
    @Label("Eps")    
    public double eps = 150;
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
                    DBScanAlgorithm.this.documents = documents;
                    DBScanAlgorithm.this.cluster(language);
                    return DBScanAlgorithm.this.clusters;
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
        
        TreeSet<Integer> unvisited = new TreeSet<Integer>();
        final List<IntArrayList> rawClusters = Lists.newArrayList();
        IntArrayList noise = new IntArrayList();
        IntArrayList neighborPts;
        IntArrayList clustered = new IntArrayList();		//punkty dodane do jakiegos klastra
        for (int i=0; i<tdMatrix.columns(); ++i) 
        	unvisited.add(i);
        while (!unvisited.isEmpty()) {
        	int i = unvisited.first();
        	unvisited.remove(i);
        	neighborPts = regionQuery(i, tdMatrix);
        	if (neighborPts.size() < minPts)
        		noise.add(i);
        	else {
        		IntArrayList newCluster = expandCluster(i, neighborPts,clustered, unvisited, tdMatrix);
        		rawClusters.add(newCluster);
        	}
        }
        for (int i = 0; i < rawClusters.size(); i++)
        {
            final Cluster cluster = new Cluster();

            final IntArrayList rawCluster = rawClusters.get(i);
            if (rawCluster.size() > 1)
            {
                cluster.addPhrases(getLabels(rawCluster,
                    vsmContext.termDocumentMatrix, rowToStemIndex,
                    preprocessingContext.allStems.mostFrequentOriginalWordIndex,
                    preprocessingContext.allWords.image));
                for (int j = 0; j < rawCluster.size(); j++)
                {
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
    
    double getDistance(DoubleMatrix1D a, DoubleMatrix1D b) {
    	double dist=0;
    	for (int i=0; i< a.size(); ++i) {
    		dist += Math.sqrt( Math.pow((a.get(i)-b.get(i)), 2) );
    	}
    	return dist;
    }
    
    IntArrayList regionQuery(int P, DoubleMatrix2D tdMatrix) {
    	IntArrayList neighbor = new IntArrayList();
    	for(int i = 0; i<tdMatrix.columns(); ++i) {
    		if(i==P)
    			neighbor.add(P);
    		if(getDistance(tdMatrix.viewColumn(i), tdMatrix.viewColumn(P)) <= eps)
    			neighbor.add(i);
    	}
    	return neighbor;
    }
    
    IntArrayList expandCluster(int P, IntArrayList neighborPts, IntArrayList clustered, TreeSet<Integer> unvisited, DoubleMatrix2D tdMatrix) {
    	IntArrayList cluster = new IntArrayList();
    	IntArrayList newNeighbor;
    	cluster.add(P);
    	clustered.add(P);
    	for (IntCursor p : neighborPts) {
    		if (unvisited.contains(p.value)) {
    			unvisited.remove(p.value);
    			newNeighbor = regionQuery(p.value, tdMatrix);
    			if (newNeighbor.size() >= minPts)
    				neighborPts.addAll(newNeighbor);
    		}
    		if (!clustered.contains(p.value)) {
    			cluster.add(p.value);
    			clustered.add(p.value);
    		}
    	}
    	return cluster;
    }
}
