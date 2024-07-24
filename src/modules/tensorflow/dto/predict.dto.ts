import { ApiProperty } from '@nestjs/swagger';

export class PredictDto {
  @ApiProperty({
    description: 'Predicción de la clase del Pokémon',
    example: 0, // Ejemplo de clase predicha
  })
  predictedClass: number;
}
