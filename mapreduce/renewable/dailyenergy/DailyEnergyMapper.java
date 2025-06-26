import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class DailyEnergyMapper extends Mapper<LongWritable, Text, Text, Text> {
    
    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ssXXX");
    private static final String[] RENEWABLE_SOURCES = {
        "biomass_generation",
        "geothermal_generation", 
        "hydro_run_of_river_and_poundage_generation",
        "hydro_water_reservoir_generation",
        "marine_generation",
        "other_renewable_generation",
        "solar_generation",
        "wind_offshore_generation",
        "wind_onshore_generation"
    };
    
    @Override
    public void map(LongWritable key, Text value, Context context) 
            throws IOException, InterruptedException {
        
        String line = value.toString();
        String[] fields = line.split(",");
        
        // Skip header row
        if (fields[0].equals("time") || fields.length < 2) {
            return;
        }
        
        try {
            // Parse timestamp
            String timestamp = fields[0].trim();
            LocalDateTime dateTime = LocalDateTime.parse(timestamp, FORMATTER);
            String dailyKey = dateTime.toLocalDate().toString();
            
            // Create energy values string
            StringBuilder energyValues = new StringBuilder();
            for (int i = 1; i <= 9 && i < fields.length; i++) {
                String energyValue = fields[i].trim();
                if (energyValue.isEmpty() || energyValue.equals("null")) {
                    energyValues.append("0.0");
                } else {
                    try {
                        double energy = Double.parseDouble(energyValue);
                        energyValues.append(String.valueOf(energy));
                    } catch (NumberFormatException e) {
                        energyValues.append("0.0");
                    }
                }
                if (i < 9) energyValues.append(",");
            }
            
            context.write(new Text(dailyKey), new Text(energyValues.toString()));
            
        } catch (Exception e) {
            context.getCounter("ERRORS", "MALFORMED_RECORDS").increment(1);
        }
    }
}