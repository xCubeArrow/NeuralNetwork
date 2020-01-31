import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Utilities {
    public static INDArray[] floatArrayToMatrixArray(float[][][] array){
        INDArray[] result = new INDArray[array.length];

        for (int i = 0; i < array.length; i++) {
            result[i] = Nd4j.create(array[i]);
        }
        return result;
    }
}
