import {
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import {
  ApiTags,
  ApiOperation,
  ApiResponse,
  ApiConsumes,
  ApiBody,
} from '@nestjs/swagger';
import { TensorflowService } from './tensorflow.service';
import { FileInterceptor } from '@nestjs/platform-express';
import { PredictDto } from './dto/predict.dto';

@ApiTags('tensorflow')
@Controller('tensorflow')
export class TensorflowController {
  constructor(private readonly tensorflowService: TensorflowService) { }

  @Post('train')
  @ApiOperation({ summary: 'Entrena el modelo de TensorFlow' })
  @ApiResponse({ status: 200, description: 'Modelo entrenado exitosamente' })
  async train(): Promise<string> {
    await this.tensorflowService.trainModel();
    return 'Training completed!';
  }

  @Post('predict')
  @ApiOperation({ summary: 'Predice la clase de un Pokémon en una imagen' })
  @ApiResponse({
    status: 200,
    description: 'Predicción de clase',
    type: PredictDto,
  })
  @ApiConsumes('multipart/form-data')
  @ApiBody({
    schema: {
      type: 'object',
      properties: {
        file: {
          type: 'string',
          format: 'binary',
        },
      },
    },
  })
  @UseInterceptors(FileInterceptor('file'))
  async predict(
    @UploadedFile() file: Express.Multer.File,
  ): Promise<PredictDto> {
    const predictedClass = await this.tensorflowService.predict(file.buffer);
    return { predictedClass };
  }
}
