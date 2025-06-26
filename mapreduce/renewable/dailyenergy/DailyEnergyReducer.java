import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class DailyEnergyReducer extends Reducer<Text, Text, Text, Text> {
    
    @Override
    public void reduce(Text key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
        
        double[] sums = new double[9];
        
        for (Text value : values) {
            String[] energyValues = value.toString().split(",");
            for (int i = 0; i < energyValues.length && i < 9; i++) {
                try {
                    sums[i] += Double.parseDouble(energyValues[i]);
                } catch (NumberFormatException e) {
                    // Skip invalid values
                }
            }
        }
        
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < 9; i++) {
            result.append(String.format("%.2f", sums[i]));
            if (i < 8) result.append(",");
        }
        
        context.write(key, new Text(result.toString()));
    }
}